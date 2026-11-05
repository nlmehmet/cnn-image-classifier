# CIFAR-10 CNN Image Classifier

**PyTorch ile görüntü sınıflandırma ve kontrollü deney karşılaştırması.**

Bu proje, CIFAR-10 veri kümesi üzerinde küçük bir evrişimli sinir ağı (CNN)
eğitir; veri artırma, Batch Normalization ve Dropout tercihlerinin modelin
başarısına etkisini karşılaştırır. Eğitimden model kaydına, deney sonuçlarının
karşılaştırılmasından hata analizine kadar komut satırından çalıştırılabilen
bir makine öğrenmesi iş akışı sunar.

Depoda kayıtlı beş deney arasında en yüksek test doğruluğu **%81.94** ile
veri artırma ve Batch Normalization kullanan `aug_bn` yapılandırmasına aittir.
Bu değer mevcut deney kayıtlarına dayanır; yeni bir eğitim çalışmasının sonucu değildir.

## Amaç ve kapsam

Amaç, aynı temel mimari üzerinde farklı eğitim tercihlerinin etkisini görünür
kılmaktır. Proje, eğitim amaçlı incelemeler ve yeni deneyler geliştirmek için
kullanılabilir.

- **Veri işleme:** CIFAR-10 indirme, normalizasyon ve isteğe bağlı veri artırma.
- **Esnek mimari:** Batch Normalization ve Dropout seçenekleri.
- **Deney takibi:** Epoch bazında kayıp, test doğruluğu ve en iyi modelin kaydı.
- **Analiz:** Karşılaştırma tablosu, eğitim eğrileri ve karışıklık matrisi.
- **Çalıştırma:** CUDA mevcutsa GPU, aksi durumda CPU desteği.

## Veri kümesi ve model

CIFAR-10, 10 sınıfa ayrılmış 32 × 32 piksel boyutunda renkli görüntülerden
oluşur. Standart ayrımda 50.000 eğitim ve 10.000 test görüntüsü bulunur.
Sınıflar: airplane, automobile, bird, cat, deer, dog, frog, horse, ship ve truck.

Model üç evrişim bloğu ve bir sınıflandırıcıdan oluşur:

| Bileşen | Yapı |
|---|---|
| Girdi | 3 × 32 × 32 RGB görüntü |
| Özellik çıkarımı | 32 → 64 → 128 kanallı üç evrişim bloğu |
| Her blok | 3 × 3 evrişim, isteğe bağlı BatchNorm, ReLU ve MaxPool |
| Sınıflandırıcı | Flatten → Dropout → Linear(2048, 256) → ReLU → Dropout → Linear(256, 10) |
| Optimizasyon | Adam ve cosine öğrenme oranı zamanlayıcısı |
| Kayıp | Cross-entropy |

Veri artırma etkin olduğunda rastgele kırpma ve yatay çevirme uygulanır.
Varsayılan eğitim ayarları 15 epoch, 128 batch boyutu, 0.001 öğrenme oranı
ve 42 seed değeridir.

## Deney sonuçları

Aşağıdaki tablo depodaki `results/*.json` kayıtlarını özetler. Doğruluk,
her deneyde eğitim boyunca gözlenen en yüksek test doğruluğudur.

| Deney | Veri artırma | BatchNorm | Dropout | En iyi test doğruluğu |
|---|---|---|---|---|
| `baseline` | Hayır | Hayır | 0.0 | %77.15 |
| `aug_drop` | Evet | Hayır | 0.3 | %77.86 |
| `aug_bn_drop` | Evet | Evet | 0.3 | %79.14 |
| `aug` | Evet | Hayır | 0.0 | %80.16 |
| `aug_bn` | Evet | Evet | 0.0 | **%81.94** |

### Bulgular

Veri artırma, temel modele göre doğruluğu **3.01 yüzde puan** artırmıştır.
Veri artırmaya Batch Normalization eklenmesiyle toplam artış **4.79 yüzde puana**
ulaşmıştır. Dropout (0.3), karşılaştırılabilir iki yapılandırmada da doğruluğu
düşürmüştür. Bu bulgu yalnızca kullanılan mimari ve eğitim ayarları için geçerlidir.

Eğitim eğrileri, temel modelde eğitim kaybı azalırken test doğruluğunun yaklaşık
%77 düzeyinde kaldığını gösterir; bu davranış aşırı öğrenmeyle uyumludur.
Veri artırma kullanılan deneylerde test başarısı yükselmiştir.

![Deneylerin eğitim kaybı ve test doğruluğu eğrileri](results/curves.png)

En iyi modelin karışıklık matrisinde özellikle cat–dog ve automobile–truck
sınıfları arasında karışıklık gözlenir. Cat sınıfında 1.000 görüntünün 613'ü
doğru sınıflandırılmış, 140'ı dog olarak tahmin edilmiştir.

![aug_bn modelinin sınıflara göre karışıklık matrisi](results/confusion_matrix.png)

### Sınırlılıklar

Deneyler tek seed ile yürütülmüştür; sonuçların değişkenliğini değerlendirmek
için birden fazla seed ile tekrar gerekir. En iyi epoch ayrı bir doğrulama kümesi
yerine test doğruluğuna göre seçildiği için raporlanan sonuçlar iyimser olabilir.
Daha sağlam bir karşılaştırma için eğitim/doğrulama ayrımı yapılmalı ve test kümesi
son değerlendirmeye ayrılmalıdır.

## Kurulum ve kullanım
Python 3.12 ile doğrulanmıştır. Proje dizininde bir sanal ortam oluşturun:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Codex Cloud ortamında hazır kurulumu kullanmak için:
```bash
source /workspace/venvs/cnn-image-classifier/bin/activate
cd /workspace/cnn-image-classifier
```

### Tek deney

Tek bir deney çalıştırın:
```bash
python src/train.py --name deneme --epochs 1 --workers 0
python src/train.py --name aug --augment
python src/train.py --help
```
CIFAR-10 ilk çalıştırmada indirilir; indirme için internet erişimi gerekir.
CUDA varsa GPU, yoksa CPU kullanılır. `--workers 0`, veriyi ana süreçte yükler.
Eğitim varsayılan olarak 15 epoch sürer. Aynı deney adıyla yeniden çalıştırmak
ilgili JSON sonucunu ve model dosyasını yeniler.

### Özel çalışma dizinleri

Veri ve çıktı konumlarını ayrı seçebilirsiniz:
```bash
python src/train.py --name deneme --epochs 1 --batch-size 64 --workers 0 \
  --seed 42 --data-dir data --results-dir runs/results --models-dir runs/models
python src/evaluate.py --results-dir runs/results --models-dir runs/models \
  --data-dir data --workers 0
```

### Değerlendirme ve görselleştirme

Depodaki kayıtlı sonuçları, model indirmeden veya eğitim yapmadan görüntülemek için:
```bash
python src/evaluate.py --curves-only --output-dir /tmp/cnn-plots
```
Tam değerlendirme için `python src/evaluate.py` çalıştırın. Bu komut, sonuçlar
arasındaki en iyi deneye ait `models/<deney>.pt` dosyasını ve CIFAR-10 verisini
ister. Model dosyaları Git deposuna dahil değildir. Grafikler varsayılan olarak
sonuç dizinine yazılır; farklı bir yer için `--output-dir` kullanın.

### Toplu deneyler

Üç deneyi sırayla eğitip değerlendirmek için:
```bash
bash run_all.sh
# Daha kısa çalıştırma:
EPOCHS=1 bash run_all.sh
```
Bu komut baseline, aug ve aug_bn_drop deneylerini `runs/` altında saklar;
depodaki eski sonuçları korur. Aynı dizinle tekrar çalıştırmak önceki çalışmanın
sonuçlarını yeniler. Yeni bir çalışma dizini için `RUN_DIR=runs/deneme2 bash run_all.sh`
kullanın. Betik herhangi bir çalışma dizininden çağrılabilir; proje dizinine geçer.

## Testler

Testler sentetik veriyle çalışır; indirme veya GPU gerektirmez. Eğitim döngüsü,
model kaydı ve yükleme, grafik üretimi, parametre doğrulama ve toplu çalıştırma
betiğinin farklı bir dizinden çağrılması kontrol edilir:

```bash
python -m unittest discover -s tests -v
```

## Proje yapısı

| Dosya / dizin | Görev |
|---|---|
| `src/data.py` | CIFAR-10 yükleme, normalizasyon ve veri artırma |
| `src/model.py` | CNN mimarisi ve yapılandırılabilir bloklar |
| `src/train.py` | Eğitim, öğrenme oranı zamanlayıcısı ve model kaydı |
| `src/evaluate.py` | Deney karşılaştırması ve görselleştirme |
| `run_all.sh` | Üç yapılandırmayı sırayla çalıştıran betik |
| `tests/` | İnternet gerektirmeyen iş akışı testleri |
| `results/` | Depoda bulunan deney kayıtları ve grafikler |
| `runs/` | Toplu çalıştırma sırasında üretilen yerel çıktılar |
| `requirements.txt` | Python bağımlılıkları |

## Geliştirme yönleri

- Ayrı doğrulama kümesi ve çoklu seed ile daha güvenilir deney karşılaştırması.
- Eğitim yapılandırmalarının genişletilmesi ve daha uzun eğitimlerin incelenmesi.
- Tek görüntü üzerinde tahmin için komut satırı aracı.

Bu maddeler geliştirme önerileridir; mevcut sürümün özellikleri değildir.
