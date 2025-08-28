# train_corrosion.py
# =========================
# BiLSTM model với đầu vào: MACCS + Physchem

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
import os
from preprocess import (
    get_mac_phys_feature,
    remove_missing_data,
    canonical_smiles,
    remove_inorganic,
    remove_mixtures,
    process_duplicate
)

# Load và tiền xử lý dữ liệu
train_df = pd.read_csv("x_train/x_train_corrosion.csv")
train_df = remove_missing_data(train_df, 'SMILES', 'Label')
train_df = canonical_smiles(train_df, 'SMILES')
train_df = remove_inorganic(train_df, 'canonical_smiles')
train_df = remove_mixtures(train_df, 'canonical_smiles')
train_df = process_duplicate(train_df, 'canonical_smiles')

# Lấy đặc trưng và label
X, y = [], []
for _, row in train_df.iterrows():
    vec = get_mac_phys_feature(row['canonical_smiles'])
    if vec is not None:
        X.append(vec)
        y.append(row['Label'])

X = np.array(X).reshape(len(X), 1, -1)  # Reshape để vào BiLSTM
y = np.array(y)

# Xây dựng mô hình BiLSTM
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, X.shape[2])))
# Tầng cuối KHÔNG return_sequences để Dense nhận vector 2D
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2)

# Lưu mô hình
os.makedirs("models", exist_ok=True)  # đảm bảo thư mục tồn tại
model.save("models/corrosion_model.keras")
