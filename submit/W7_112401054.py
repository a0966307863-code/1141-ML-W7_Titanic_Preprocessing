# -*- coding: utf-8 -*-
# W6 Titanic Preprocessing Template
# 僅可修改 TODO 區塊，其餘部分請勿更動

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 任務 1：載入資料
def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    df = pd.read_csv(file_path)
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    # TODO 2.2: 以 Embarked 眾數填補
    # .mode()[0] 取出眾數
    if 'Embarked' in df.columns:
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'].fillna(embarked_mode, inplace=True)
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # 確保 'Fare' 欄位存在且為數值型
    if 'Fare' not in df.columns or not pd.api.types.is_numeric_dtype(df['Fare']):
        return df
    
    # 複製 DataFrame 以便在迴圈中操作
    df_cleaned = df.copy()
    
    # 初始化一個變數來儲存前一次的資料筆數
    previous_len = 0
    current_len = len(df_cleaned)
    
    # 使用 while 迴圈進行迭代移除，直到資料筆數不再變化
    while current_len != previous_len:
        previous_len = current_len
        
        # TODO 3.1: 計算 Fare 平均與標準差 (基於當前的 df_cleaned)
        # 確保只對非缺失值計算統計量
        fare_data = df_cleaned['Fare'].dropna()
        fare_mean = fare_data.mean()
        fare_std = fare_data.std()
        
        # 設定門檻：平均值 + 3 * 標準差 (單側異常值移除)
        threshold = fare_mean + 3 * fare_std
        
        # TODO 3.2: 移除 Fare > mean + 3*std 的資料
        # 使用布林索引保留「非」異常值
        df_cleaned = df_cleaned[df_cleaned['Fare'] <= threshold].copy()
        
        # 更新當前資料筆數
        current_len = len(df_cleaned)
        
    # 回傳最終清理後的 DataFrame
    return df_cleaned


# 任務 4：類別變數編碼 (已修正: 移除 drop_first=True)
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    # 根據錯誤，必須保留所有虛擬變數
    cols_to_encode = [col for col in ['Sex', 'Embarked'] if col in df.columns]
    
    # 移除 drop_first=True
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    
    features_to_scale = [col for col in ['Age', 'Fare'] if col in df.columns]
    
    df_scaled = df.copy()
    
    if features_to_scale:
        df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
        
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # 確保 'Survived' 欄位存在
    if 'Survived' not in df.columns:
        # 如果 'Survived' 不存在，返回空的切割
        return None, None, None, None

    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    X = df.drop('Survived', axis=1) 
    y = df['Survived']
    
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")


# 主程式流程（請勿修改）
if __name__ == "__main__":
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    # 這裡的程式碼已經完成了完整的數據預處理流程
    df, missing_count = load_data(input_path)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(df, output_path)

    print("Titanic 資料前處理完成")