import pandas as pd

veriler=pd.read_csv('breast.csv')

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

#string değerleri sayısal ile değiştir
veriler['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True) #M kötü huylu,B iyi huylu

#conqure datas for test and train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

independent_variables = veriler.drop(columns=["diagnosis"])
dependent_variables = veriler.iloc[:, 1]

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
selected_features = veriler[['points_worst', 'perimeter_worst', 'radius_worst', 'points_mean', 'area_worst']]     

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
points_worst = float(input(" points_worst değerini giriniz: "))
perimeter_worst = float(input(" perimeter_worst değerini giriniz: "))
radius_worst = float(input(" radius_worst değerini giriniz: "))
points_mean = float(input(" points_mean değerini giriniz: "))
area_worst = float(input("area_worst değerini giriniz: "))

# Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
kullanici_verisi = pd.DataFrame({'points_worst': [points_worst], 'perimeter_worst': [perimeter_worst], 'radius_worst': [radius_worst], 'points_mean': [points_mean], 'area_worst': [area_worst]})

# Model kullanıcı verisini kullanarak tahmin yap
tahmin = model.predict(kullanici_verisi)

if tahmin==1 :
 print("kötü huylu tümör")
elif tahmin==0:
 print("iyi huylu tümör")   
 






