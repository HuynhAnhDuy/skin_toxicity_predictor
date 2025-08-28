# train_sensitization.py
# =========================
# BiLSTM với đầu vào: Token SMILES + RDKit fingerprint (kết hợp)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tokenizer import CustomTokenizer
import os
from preprocess import (
    get_token_fp_feature,
    remove_missing_data,
    canonical_smiles,
    remove_inorganic,
    remove_mixtures,
    process_duplicate
)

# 1) Tiền xử lý dữ liệu
train_df = pd.read_csv("x_train/x_train_sensitization.csv")

train_df = remove_missing_data(train_df, 'SMILES', 'Label')
train_df = canonical_smiles(train_df, 'SMILES')
train_df = remove_inorganic(train_df, 'canonical_smiles')
train_df = remove_mixtures(train_df, 'canonical_smiles')
train_df = process_duplicate(train_df, 'canonical_smiles')
tokenizer = CustomTokenizer()
# 2) Tạo đặc trưng kết hợp Token + RDKit (vector 1 chiều)
X, y = [], []
for _, row in train_df.iterrows():
    vec = get_token_fp_feature(row['canonical_smiles'],tokenizer)  # np.array shape: (max_len + nBits,)
    if vec is not None:
        X.append(vec)
        y.append(row['Label'])

X = np.array(X).reshape(len(X), 1, -1)   # (samples, timesteps=1, features)
y = np.array(y)

print("X shape:", X.shape, " y shape:", y.shape)  # ví dụ: (N, 1, 2148)

# 3) Xây dựng mô hình BiLSTM
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, X.shape[2])))
# Tầng cuối KHÔNG return_sequences để Dense nhận vector 2D
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4) Huấn luyện (kèm EarlyStopping)
model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

# 5) Lưu mô hình
os.makedirs("models", exist_ok=True)
model.save("models/sensitization_model.keras")
print("✓ Saved to sensitization_model.keras")
