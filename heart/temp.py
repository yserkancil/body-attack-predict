import clr
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clr.AddReference("System.Windows.Forms")

# Veri setindeki null değerleri kontrol eden ve gerekli olanları sütun ortalaması ile dolduran fonksiyon
def null_kontrol_doldur(veri_seti):
    for i in range(veri_seti.shape[1]):
        null_values = np.isnan(veri_seti[:, i])
        if np.any(null_values):
            mean_value = np.nanmean(veri_seti[:, i])
            veri_seti[null_values, i] = mean_value
    return veri_seti

# Modeli eğit ve kaydet
def modeli_egit_kaydet():
    # Veri setini yükle
    veriler = np.genfromtxt('heart.csv', delimiter=',', skip_header=1)
    
    # Sadece float64 ve int64 tipindeki verileri seç
    veriler = veriler[:, [i for i in range(veriler.shape[1]) if veriler[:, i].dtype in [np.float64, np.int64]]]

    # Null değerleri kontrol et ve doldur
    veriler = null_kontrol_doldur(veriler)

    # Bağımsız ve bağımlı değişkenleri ayır
    independent_variables = veriler[:, :13]
    dependent_variables = veriler[:, 13]

    # Verileri eğitim ve test setlerine böl
    independent_variables_train, independent_variables_test, dependent_variables_train, dependent_variables_test = train_test_split(
        independent_variables, dependent_variables, test_size=0.33, random_state=0)

    # Modeli seç ve eğit
    model = RandomForestClassifier(random_state=0)
    model.fit(independent_variables_train, dependent_variables_train)

    # Modeli test verisiyle değerlendir
    dependent_variables_pred = model.predict(independent_variables_test)
    accuracy = accuracy_score(dependent_variables_test, dependent_variables_pred)
    print("Modelin doğruluk skoru:", accuracy)

    # Özellik önemliliği değerlerini al
    feature_importances = model.feature_importances_

    # Özelliklerin indekslerini ve önemlilik değerlerini birleştir
    feature_importances_list = list(zip(range(independent_variables.shape[1]), feature_importances))

    # Özellikleri önemlilik değerlerine göre sırala
    sorted_features = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)

    # En önemli ilk beş özelliği yazdır
    print("En önemli ilk beş özellik:")
    for feature, importance_score in sorted_features[:5]:
        print(f"Özellik: {feature}, Önemlilik Değeri: {importance_score:.4f}")

    # En önemli beş özelliği seç
    selected_columns = [feature for feature, _ in sorted_features[:5]]
    selected_features = independent_variables[:, selected_columns]

    X_train, X_test, y_train, y_test = train_test_split(selected_features, dependent_variables, test_size=0.25, random_state=0)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Seçilen özelliklerle modelin doğruluk skoru:", accuracy)

    # Modeli yeniden kaydet (seçilen özelliklerle)
    joblib.dump(model, 'selected_features_heart_model.pkl')

def tahmin_yap():
    # Kaydedilen modeli yükle
    model = joblib.load('selected_features_heart_model.pkl')

    # Kullanıcıdan beş özelliği girmesini iste
    oldpeak = float(raw_input("oldpeak değerini giriniz: "))
    thal = float(raw_input("thal değerini giriniz: "))
    cp = float(raw_input("cp değerini giriniz: "))
    thalach = float(raw_input("thalach değerini giriniz: "))
    ca = float(raw_input("ca değerini giriniz: "))

    # Kullanıcının girdiği değerlerle bir numpy array oluştur
    kullanici_verisi = np.array([[oldpeak, thal, cp, thalach, ca]])

    # Model kullanıcı verisini kullanarak tahmin yap
    tahmin = model.predict(kullanici_verisi)

    # Tahmini yazdır
    if tahmin[0] == 1:
        print("Kalp krizi riski var.")
    else:
        print("Kalp krizi riski yok.")

if __name__ == "__main__":
    modeli_egit_kaydet()  # Modeli eğit ve kaydet
    tah
