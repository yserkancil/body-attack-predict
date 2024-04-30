import pandas as pd

veriler=pd.read_csv('parkinson.csv')

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

veriler = veriler.drop(columns=["name"])
#conqure datas for test and train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Bağımlı değişken (dependent variable) için 'state' sütununu seç
dependent_variables = veriler['status']

# Bağımsız değişkenler (independent variables) için 'state' sütununu dışarıda bırakarak diğer sütunları seç
independent_variables = veriler.drop(columns=["status"])

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
selected_features = veriler[['spread1', 'PPE', 'MDVP_Fo_Hz', 'spread2', 'MDVP_Fhi_Hz']] 
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
spread1 = float(input(" spread1 değerini giriniz: "))
PPE = float(input(" PPE değerini giriniz: "))
MDVP_Fo_Hz = float(input(" MDVP_Fo_Hz değerini giriniz: "))
spread2 = float(input(" spread2 değerini giriniz: "))
MDVP_Fhi_Hz = float(input("MDVP_Fhi_Hz değerini giriniz: "))

# Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
kullanici_verisi = pd.DataFrame({'spread1': [spread1], 'PPE': [PPE], 'MDVP_Fo_Hz': [MDVP_Fo_Hz], 'spread2': [spread2], 'MDVP_Fhi_Hz': [MDVP_Fhi_Hz]})

# Model kullanıcı verisini kullanarak tahmin yap
tahmin = model.predict(kullanici_verisi)

if tahmin == 1:
    print("parkinson")
elif tahmin == 0:
    print("parkinson değil")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    