import os
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_log_mel_spectrogram(file_path, sr=None, frame_size=0.064, hop_size=0.032, n_mels=128):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(sr * frame_size),
                                              hop_length=int(sr * hop_size), n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec)
    return log_mel


def create_feature_windows(log_mel, context_frames=5):
    frames = []
    for i in range(len(log_mel[0]) - context_frames + 1):
        window = log_mel[:, i:i+context_frames].reshape(-1)
        frames.append(window)
    print(f"Feature Windows: {len(frames)}")
    return np.array(frames)


def normalize_data(data, method="minmax"):
    """
    데이터를 정규화 또는 표준화
    Args:
        data (np.array): 입력 데이터 (2차원 배열)
        method (str): "minmax" 또는 "standard". 기본값은 "minmax".
    Returns:
        normalized_data (np.array): 정규화 또는 표준화된 데이터.
    """
    if method == "minmax":
        print("Applying MinMax Normalization...")
        scaler = MinMaxScaler()
    elif method == "standard":
        print("Applying Standardization...")
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Choose 'minmax' or 'standard'.")

    return scaler.fit_transform(data)




def load_data(normal_dir, abnormal_dir, context_frames=5, normalize_method=None):

    """
    정상 및 비정상 데이터 로드 및 전처리 (옵션으로 정규화 추가)
    Args:
        normal_dir (str): 정상 데이터 경로
        abnormal_dir (str): 비정상 데이터 경로
        context_frames (int): 연속된 프레임 수
        normalize_method (str): "minmax" 또는 "standard" 중 선택 (정규화 방식, 없으면 None)
    Returns:
        normal_data (np.array): 정규화된 정상 데이터
        abnormal_data (np.array): 정규화된 비정상 데이터
    """

    normal_data, abnormal_data = [], []

    # Load normal data
    print("Loading normal data...")
    for root, _, files in os.walk(normal_dir):
        for file in files:
            if file.endswith(".wav"):
                log_mel = extract_log_mel_spectrogram(os.path.join(root, file))
                feature_windows1 = create_feature_windows(log_mel, context_frames)
                print(f"File: {file}, Total Feature Windows: {len(feature_windows1)}")  # Feature 개수 출력
                normal_data.extend(feature_windows1)

    # Load abnormal data
    print("Loading abnormal data...")
    for root, _, files in os.walk(abnormal_dir):
        for file in files:
            if file.endswith(".wav"):
                log_mel = extract_log_mel_spectrogram(os.path.join(root, file))
                feature_windows2 = create_feature_windows(log_mel, context_frames)
                print(f"File: {file}, Total Feature Windows: {len(feature_windows2)}")
                abnormal_data.extend(feature_windows2)


    normal_data = np.array(normal_data)
    abnormal_data = np.array(abnormal_data)

    # Apply normalization if specified
    if normalize_method:
        print(f"Applying {normalize_method} normalization to normal and abnormal data...")
        normal_data = normalize_data(normal_data, method=normalize_method)
        abnormal_data = normalize_data(abnormal_data, method=normalize_method)

    print(f"Normal data shape: {normal_data.shape}, Abnormal data shape: {abnormal_data.shape}")
    return normal_data, abnormal_data