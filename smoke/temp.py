import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def null_kontrol_doldur(veri_seti):
    null_degerler = veri_seti.isnull().sum()
    for sutun in veri_seti.columns:
        if null_degerler[sutun] > 0:
            sutun_ort = veri_seti[sutun].mean()
            veri_seti[sutun].fillna(sutun_ort, inplace=True)
    return veri_seti

def modeli_egit_kaydet():
    # Veri setini yükle
    veriler = pd.read_csv('smoking.csv')

    veriler = null_kontrol_doldur(veriler)

    veriler['gender'].replace({'F': 1, 'M': 0}, inplace=True)
    veriler['tartar'].replace({'Y': 1, 'N': 0}, inplace=True)
    veriler.drop(columns=["oral"], inplace=True)

    independent_variables = veriler.iloc[:, :25]
    dependent_variables = veriler.iloc[:, 25]

    independent_variables_train, independent_variables_test, dependent_variables_train, dependent_variables_test = train_test_split(independent_variables, dependent_variables, test_size=0.33, random_state=0)

    model = RandomForestClassifier(random_state=0)
    model.fit(independent_variables_train, dependent_variables_train)

    dependent_variables_pred = model.predict(independent_variables_test)
    accuracy = accuracy_score(dependent_variables_test, dependent_variables_pred)
    print("Modelin doğruluk skoru:", accuracy)

    # Seçilen özelliklerin olduğu yeni veri seti
    selected_features = veriler[['gender', 'hemoglobin', 'Gtp', 'triglyceride', 'height']]  

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

    # Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
    kullanici_verisi = pd.DataFrame({'gender': [gender], 'hemoglobin': [hemoglobin], 'Gtp': [Gtp], 'triglyceride': [triglyceride], 'height': [height]})

    # Model kullanıcı verisini kullanarak tahmin yap
    tahmin = model.predict(kullanici_verisi)

    if tahmin == 1:
        print("sigara içiyor")
    else:
        print("sigara içmiyor")

if __name__ == "__main__":
    modeli_egit_kaydet()  # Modeli eğit ve kaydet
    tahmin_yap()          # Kullanıcıdan değerleri al ve tahmin yap
