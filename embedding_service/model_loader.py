import onnxruntime as ort

def create_session(model_path: str):
    providers = [
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
    session = ort.InferenceSession(
        model_path,
        providers=providers,
        provider_options=None
    )
    print(f"[INFO] ONNX provider: {session.get_providers()[0]}")
    return session