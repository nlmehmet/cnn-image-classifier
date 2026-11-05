# CIFAR-10 CNN Image Classifier

PyTorch ile CIFAR-10 üzerinde küçük bir CNN eğitip üç farklı ayarı karşılaştıran proje:
veri artırma (augmentation), BatchNorm ve Dropout'un doğruluğa etkisi.

## Kurulum ve çalıştırma
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

Veri ve çıktı konumlarını ayrı seçebilirsiniz:
```bash
python src/train.py --name deneme --epochs 1 --batch-size 64 --workers 0 \
  --seed 42 --data-dir data --results-dir runs/results --models-dir runs/models
python src/evaluate.py --results-dir runs/results --models-dir runs/models \
  --data-dir data --workers 0
```

Depodaki kayıtlı sonuçları, model indirmeden veya eğitim yapmadan görüntülemek için:
```bash
python src/evaluate.py --curves-only --output-dir /tmp/cnn-plots
```
Tam değerlendirme için `python src/evaluate.py` çalıştırın. Bu komut, sonuçlar
arasındaki en iyi deneye ait `models/<deney>.pt` dosyasını ve CIFAR-10 verisini
ister. Model dosyaları Git deposuna dahil değildir. Grafikler varsayılan olarak
sonuç dizinine yazılır; farklı bir yer için `--output-dir` kullanın.

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

Testler (indirme veya GPU gerektirmez):
```bash
python -m unittest discover -s tests -v
```

## Sonuçlar
> `python src/evaluate.py` çıktısındaki tabloyu buraya yapıştır.

| Deney | Augment | BatchNorm | Dropout | En iyi doğruluk |
|---|---|---|---|---|
| baseline | False | False | 0.0 | 77.15% |
| aug_drop | True | False | 0.3 | 77.86% |
| aug_bn_drop | True | True | 0.3 | 79.14% |
| aug | True | False | 0.0 | 80.16% |
| aug_bn | True | True | 0.0 | 81.94% |


![Eğitim eğrileri](results/curves.png)
![Confusion matrix](results/confusion_matrix.png)

## Yorum
Beş ayar karşılaştırıldı (ablation). Veri artırma (RandomCrop + yatay çevirme) test doğruluğunu %77.15'ten %80.16'ya çıkardı (+3.01 puan). BatchNorm eklenmesi iki karşılaştırmada da fayda sağladı ve en iyi sonucu veren aug_bn modeline ulaştı: %81.94. Dropout (0.3) ise her kombinasyonda doğruluğu 2-3 puan düşürdü.

Eğri grafikleri bunu açıklıyor: baseline modelinde eğitim kaybı yaklaşık 0.04'e inerken test doğruluğu 5. epoch civarından itibaren %77 civarında kaldı; bu belirgin bir overfitting işaretidir. Veri artırma eğitim kaybını yüksek tutarak ezberlemeyi azalttı. Dropout'lu modellerde ise eğitim kaybı en yüksek seviyede (~0.70) kaldı ve test doğruluğu daha yavaş yükseldi; model küçük ve eğitim 15 epoch ile sınırlı olduğu için Dropout burada düzenleme yerine eğitimi yavaşlatan bir etki yapmış olabilir.

Confusion matrix'te (aug_bn) en zayıf sınıf cat oldu (1000 örnekten 613 doğru); 140'ı dog olarak tahmin edildi. Dog sınıfında da 121 örnek cat ile karıştırıldı. Automobile–truck çifti de küçük bir karışıklık gösterdi.

Sınırlılıklar: Tüm deneyler tek seed ile çalıştırıldı; sonuçlar yön olarak tutarlı olsa da kesinleştirmek için çoklu seed ile tekrar gerekir. Ayrıca en iyi epoch, ayrı bir doğrulama kümesi olmadan test doğruluğuna göre seçildi, bu yüzden raporlanan değerler hafif iyimser olabilir.

## Proje yapısı
- `src/data.py` – veri yükleme ve augmentation
- `src/model.py` – CNN mimarisi
- `src/train.py` – eğitim döngüsü, cosine LR scheduler, en iyi modeli kaydetme
- `src/evaluate.py` – karşılaştırma tablosu, eğri grafiği, confusion matrix
