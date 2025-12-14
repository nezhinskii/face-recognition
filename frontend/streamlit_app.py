import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="Face Recognition Demo", layout="centered")
st.title("🎭 Face Recognition API Demo")

API_BASE_URL = st.text_input("API Base URL", value="http://face-api:8000", key="api_url")

def draw_boxes(image_bytes: bytes, detections: list, best_det_id: int | None = None) -> Image.Image:
    """Рисует bounding boxes и уверенность на изображении"""
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    
    for idx, det in enumerate(detections):
        box = det["bbox"]
        confidence = det["conf"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[0] - 20), f"{confidence:.2f}", fill="red")

        if best_det_id is not None and idx == best_det_id:
            draw.rectangle(box, outline="lime", width=5)
            draw.text((box[0], box[0] - 40), "BEST", fill="lime")
    
    return image

def upload_image(key: str) -> tuple[bytes, Image.Image | None]:
    uploaded_file = st.file_uploader("Загрузите фото", type=["jpg", "jpeg", "png"], key=key)
    if uploaded_file:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image_bytes, image
    return None, None

st.sidebar.header("Выберите действие")
action = st.sidebar.radio("Действие", 
                          ["Добавить нового человека", 
                           "Распознать человека", 
                           "Удалить человека"])

if action == "Добавить нового человека":
    st.header("➕ Добавление нового человека")
    
    name = st.text_input("Имя человека")
    image_bytes, preview_image = upload_image("add_person")
    
    if st.button("Добавить") and name and image_bytes:
        with st.spinner("Отправка..."):
            files = {"file": ("photo.jpg", image_bytes, "image/jpeg")}
            data = {"name": name}
            
            response = requests.post(f"{API_BASE_URL}/api/new_person", files=files, data=data)
        
        if response.status_code == 201:
            result = response.json()
            st.success(f"Успешно добавлен человек: **{result['name']}** (ID: {result['id']})")
            st.write(f"Обнаружено лиц: {result['faces_detected']}")
            
            # Сохраняем для отрисовки
            st.session_state.last_response = result
            drawn = draw_boxes(image_bytes, result["detections"], result["best_det_id"])
            st.image(drawn, caption="Обнаруженные лица (зелёный — лучшее)")
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            st.error(f"Ошибка: {response.status_code} — {error_detail}")

elif action == "Распознать человека":
    st.header("🔍 Распознавание человека на фото")
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Порог сходства", min_value=0.0, max_value=1.0, 
                              value=0.35, step=0.01, 
                              help="Чем выше — тем строже поиск")
    
    image_bytes, preview_image = upload_image("recognize")
    
    if st.button("Распознать") and image_bytes:
        with st.spinner("Поиск..."):
            files = {"file": ("photo.jpg", image_bytes, "image/jpeg")}
            data = {"threshold": threshold}
            
            response = requests.post(f"{API_BASE_URL}/api/get_person", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            similarity = result.get("similarity", 0)
            st.success(f"Найден человек: **{result['name']}** (ID: {result['id']})")
            st.metric("Сходство", f"{similarity:.4f}")
            st.write(f"Обнаружено лиц: {result['faces_detected']}")
            
            st.session_state.last_response = result
            drawn = draw_boxes(image_bytes, result["detections"], result["best_det_id"])
            st.image(drawn, caption="Обнаруженные лица (зелёный — лучшее, использованное для поиска)")
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            st.error(f"Ошибка: {response.status_code} — {error_detail}")

else:
    st.header("🗑 Удаление человека")
    
    person_id = st.number_input("ID человека для удаления", min_value=1, step=1)
    
    if st.button("Удалить", type="primary"):
        with st.spinner("Удаление..."):
            data = {"id": person_id}
            response = requests.delete(f"{API_BASE_URL}/api/delete_person", data=data)
        
        if response.status_code == 204:
            st.success(f"Человек с ID {person_id} успешно удалён")
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            st.error(f"Ошибка: {response.status_code} — {error_detail}")

with st.expander("🔍 Проверить статус API"):
    if st.button("Health check"):
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                st.success("API работает ✅")
                st.json(resp.json())
            else:
                st.error(f"API вернул {resp.status_code}")
        except Exception as e:
            st.error(f"Не удалось подключиться: {e}")