import pandas as pd
import re

veriler=pd.read_csv('mental.csv')

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

#Değerleri sayısal verilerle değiştir 
veriler['Patient Number'] = veriler['Patient Number'].apply(lambda x: re.search(r'\d+$', x).group())
veriler['Sadness'].replace({'Usually': 1, 'Sometimes': 2, 'Seldom': 3, 'Most-Often': 4}, inplace=True)
veriler['Euphoric'].replace({'Usually': 1, 'Sometimes': 2, 'Seldom': 3, 'Most-Often': 4}, inplace=True)
veriler['Exhausted'].replace({'Usually': 1, 'Sometimes': 2, 'Seldom': 3, 'Most-Often': 4}, inplace=True)
veriler['Sleep dissorder'].replace({'Usually': 1, 'Sometimes': 2, 'Seldom': 3, 'Most-Often': 4}, inplace=True)
veriler['Mood Swing'].replace({'YES': 1, 'NO': 0}, inplace=True)
sil_index = 0  #hataolduğu için 1. değeri sil
veriler.drop(sil_index, inplace=True)
veriler['Suicidal thoughts'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Anorxia'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Authority Respect'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Try-Explanation'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Aggressive Response'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Ignore & Move-On'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Nervous Break-down'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Admit Mistakes'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Overthinking'].replace({'YES': 1, 'NO': 0}, inplace=True)
veriler['Sexual Activity'] = veriler['Sexual Activity'].apply(lambda x: re.match(r'\d+', x).group())
veriler['Concentration'] = veriler['Concentration'].apply(lambda x: re.match(r'\d+', x).group())
veriler['Optimisim'] = veriler['Optimisim'].apply(lambda x: re.match(r'\d+', x).group())
veriler['Expert Diagnose'].replace({'Bipolar Type-1': 1, 'Bipolar Type-2': 2, 'Normal': 3, 'Depression': 4}, inplace=True)

#conqure datas for test and train
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

independent_variables = veriler.iloc[:, :18]
dependent_variables = veriler.iloc[:, 18]

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
selected_features = veriler[['Mood Swing', 'Optimisim', 'Sexual Activity', 'Euphoric', 'Concentration']]  

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
Mood_Swing = float(input(" Mood_Swing değerini giriniz: "))
Optimisim = float(input(" Optimisim değerini giriniz: "))
Sexual_Activity = float(input(" Sexual_Activity değerini giriniz: "))
Euphoric = float(input(" Euphoric değerini giriniz: "))
Concentration = float(input("Concentration değerini giriniz: "))

# Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
kullanici_verisi = pd.DataFrame({'Mood Swing': [Mood_Swing], 'Optimisim': [Optimisim], 'Sexual Activity': [Sexual_Activity], 'Euphoric': [Euphoric], 'Concentration': [Concentration]})

# Model kullanıcı verisini kullanarak tahmin yap
tahmin = model.predict(kullanici_verisi)

print(tahmin)













  
