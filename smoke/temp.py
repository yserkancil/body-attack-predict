import pandas as pd

veriler=pd.read_csv('smoking.csv')

# Veri setindeki null değerleri kontrol eden ve gerekli olanları sütun ortalaması ile dolduran fonksiyon
def null_kontrol_doldur(veri_seti):
    # Veri setindeki null değerlerin toplam sayısını al
    null_degerler = veri_seti.isnull().sum()
    # Her sütunu kontrol et
    for sutun in veri_seti.columns:
        if null_degerler[sutun] > 0:
            sutun_ort = veri_seti[sutun].mean()
            veri_seti[sutun].fillna(sutun_ort, inplace=True)
        return veri_seti
veriler = null_kontrol_doldur(veriler)

veriler['gender'].replace({'F': 1, 'M': 0}, inplace=True)
veriler['tartar'].replace({'Y': 1, 'N': 0}, inplace=True)
veriler.drop(columns=["oral"], inplace=True)

#conqure datas for test and train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

independent_variables = veriler.iloc[:, :25]
dependent_variables = veriler.iloc[:, 25]

independent_variables_train,independent_variables_test,dependent_variables_train,dependent_variables_test = train_test_split(independent_variables,dependent_variables,test_size=0.33 ,random_state=(0))

# Modeli seç ve eğit
model = RandomForestClassifier(random_state=0)
model.fit(independent_variables_train, dependent_variables_train)

# Modeli test verisiyle değerlendir
dependent_variables_pred = model.predict(independent_variables_test)
accuracy = accuracy_score( dependent_variables_test, dependent_variables_pred)
print("Modelin doğruluk skoru:", accuracy)

# Özellik önemliliği değerlerini al
feature_importances = model.feature_importances_
# Özelliklerin indekslerini ve önemlilik değerlerini birleştir
feature_importances_list = list(zip(independent_variables.columns, feature_importances))

# Özellikleri önemlilik değerlerine göre sırala
sorted_features = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)

# En önemli ilk beş özelliği yazdır
print("En önemli ilk beş özellik:")
for feature, importance_score in sorted_features[:5]:
    print(f"Özellik: {feature}, Önemlilik Değeri: {importance_score:.4f}")
    
# İlgili özelliklerin bulunduğu veri setini oluştur
selected_features = veriler[['gender', 'hemoglobin', 'Gtp', 'triglyceride', 'height']]  

# Verileri eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(selected_features, dependent_variables, test_size=0.33, random_state=0)
# Yeni bir rastgele orman modeli oluştur
model = RandomForestClassifier(random_state=0)

# Modeli eğit
model.fit(X_train, y_train)

# Modelin tahminlerini yap
y_pred = model.predict(X_test)

# Doğruluk değerini hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluk skoru:", accuracy)

# Kullanıcıdan beş özelliği girmesini iste
gender= float(input(" gender değerini giriniz: "))
hemoglobin = float(input(" hemoglobin değerini giriniz: "))
Gtp = float(input(" Gtp değerini giriniz: "))
triglyceride = float(input(" triglyceride değerini giriniz: "))
height = float(input("height değerini giriniz: "))

# Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
kullanici_verisi = pd.DataFrame({'gender': [gender], 'hemoglobin': [hemoglobin], 'Gtp': [Gtp], 'triglyceride': [triglyceride], 'height': [height]})

# Model kullanıcı verisini kullanarak tahmin yap
tahmin = model.predict(kullanici_verisi)

print(tahmin)    
if tahmin == 1:
    print("sigara içiyor")
elif tahmin == 0:
    print("sigara içmiyor")