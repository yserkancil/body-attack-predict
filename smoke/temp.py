import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def null_kontrol_doldur(veri_seti):
    for sutun in range(veri_seti.shape[1]):
        if np.isnan(veri_seti[:, sutun]).any():
            sutun_ort = np.nanmean(veri_seti[:, sutun])
            veri_seti[np.isnan(veri_seti[:, sutun]), sutun] = sutun_ort
    return veri_seti

def modeli_egit_kaydet():
    # Veri setini yükle
    veriler = np.genfromtxt('smoking.csv', delimiter=',', skip_header=1)

    veriler = null_kontrol_doldur(veriler)

    # 'gender' sütunu: 'F' = 1, 'M' = 0
    veriler[:, 1] = np.where(veriler[:, 1] == 'F', 1, 0)

    # 'tartar' sütunu: 'Y' = 1, 'N' = 0
    veriler[:, -1] = np.where(veriler[:, -1] == 'Y', 1, 0)

    # 'oral' sütununu çıkar
    veriler = np.delete(veriler, -2, axis=1)

    independent_variables = veriler[:, :25]
    dependent_variables = veriler[:, 25]

    independent_variables_train, independent_variables_test, dependent_variables_train, dependent_variables_test = train_test_split(independent_variables, dependent_variables, test_size=0.33, random_state=0)

    model = RandomForestClassifier(random_state=0)
    model.fit(independent_variables_train, dependent_variables_train)

    dependent_variables_pred = model.predict(independent_variables_test)
    accuracy = accuracy_score(dependent_variables_test, dependent_variables_pred)
    print("Modelin doğruluk skoru:", accuracy)

    # Seçilen özelliklerin olduğu yeni veri seti
    selected_features = veriler[:, [1, 7, 12, 15, 5]]

    X_train, X_test, y_train, y_test = train_test_split(selected_features, dependent_variables, test_size=0.33, random_state=0)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Seçilen özelliklerle modelin doğruluk skoru:", accuracy)

    # Modeli yeniden kaydet (seçilen özelliklerle)
    joblib.dump(model, 'selected_features_smoking_model.pkl')

def tahmin_yap():
    # Kaydedilen modeli yükle
    model = joblib.load('selected_features_smoking_model.pkl')

    # Kullanıcıdan beş özelliği girmesini iste
    gender = float(input("gender değerini giriniz (1: Kadın, 0: Erkek): "))
    hemoglobin = float(input("hemoglobin değerini giriniz: "))
    Gtp = float(input("Gtp değerini giriniz: "))
    triglyceride = float(input("triglyceride değerini giriniz: "))
    height = float(input("height değerini giriniz: "))

    # Kullanıcının girdiği değerlerle bir veri matrisi oluştur
    kullanici_verisi = np.array([[gender, hemoglobin, Gtp, triglyceride, height]])

    # Model kullanıcı verisini kullanarak tahmin yap
    tahmin = model.predict(kullanici_verisi)

    if tahmin == 1:
        print("sigara içiyor")
    else:
        print("sigara içmiyor")

if __name__ == "__main__":
    modeli_egit_kaydet()  # Modeli eğit ve kaydet
    tahmin_yap()          # Kullanıcıdan değerleri al ve tahmin yap

