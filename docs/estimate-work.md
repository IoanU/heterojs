# Evaluarea estimării de cost în HeteroJS

Au fost verificate **3.280 probe cronometrate** pe AMD Ryzen 7 5800H with Radeon Graphics, Windows 10.0.26200. Rezultatul fiecărei probe a fost comparat element cu element cu implementarea CPU. Câmpul de cost este activat opțional, prin `plan.estimateWork`.

## 1. Efectul estimării de cost

Cu 524.288 de elemente pe Radeon, estimarea a redus mediana de la **21,35 la 10,90 ms** când zona scumpă era fixă și de la **31,45 la 13,05 ms** când zona grea se muta între apeluri. În aceste două cazuri, `adaptive-work` a depășit `cpu-only`, `gpu-only` și `static` la mediană și p95.

Intrările uniforme au avut o mediană mai mare cu estimarea. La 524.288 de elemente, numărul median de segmente a crescut de la 20 la 29. Pentru lucrările uniforme, costul uniform per element și segmentele mai mari pot fi suficiente.

### Chrome pe Radeon, fără încărcare artificială

Fiecare celulă conține mediana și p95 în milisecunde, după încălzire.

| Distribuție | Elemente | `cpu-only` | `gpu-only` | `static` | `adaptive` | `adaptive-work` |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 65.536 | 4,70 / 5,10 | 4,60 / 9,40 | 4,50 / 6,00 | 5,40 / 6,00 | 5,90 / 6,30 |
| Uniform | 524.288 | 31,50 / 32,40 | 19,40 / 21,70 | 13,90 / 16,00 | 18,80 / 21,10 | 20,65 / 25,80 |
| Zonă grea fixă | 65.536 | 13,80 / 14,30 | 4,90 / 5,60 | 4,80 / 6,30 | 5,90 / 14,40 | 5,75 / 6,50 |
| Zonă grea fixă | 524.288 | 106,30 / 107,70 | 20,20 / 22,00 | 30,05 / 47,40 | 21,35 / 67,00 | 10,90 / 13,40 |
| Zonă grea mobilă | 65.536 | 13,95 / 14,50 | 4,80 / 5,80 | 4,80 / 6,00 | 8,95 / 15,00 | 5,95 / 8,50 |
| Zonă grea mobilă | 524.288 | 106,55 / 108,70 | 19,50 / 21,90 | 23,30 / 46,70 | 31,45 / 92,30 | 13,05 / 17,30 |

### Metoda comparației pereche

- Am folosit același kernel CPU și același kernel WGSL în toate strategiile. Fiecare element parcurge o recurență întreagă pe 32 de biți. Cazul uniform are 70 de iterații per element. Cazurile neuniforme au 8 iterații în zona ușoară și 256 în sfertul greu. Cazul mobil mută zona grea de la începutul la sfârșitul tabloului la apeluri succesive.
- Cele două variante adaptive folosesc același scheduler pe segmente. `adaptive-work` primește ca indiciu suma numărului de iterații pe interval. Fiecare mod are propriul motor cu patru workeri CPU. Strategia statică începe cu 25% dintre elemente alocate CPU-ului. Ordinea modurilor se rotește între probe, iar intrarea este identică în interiorul fiecărui grup de probe.
- Cronometrul pornește la apelul `compute` și se oprește la rezultat. Verificarea valorilor este în afara cronometrării. Datele brute conțin primul apel, încălzirile și probele, cu alocarea CPU/GPU, costul estimat și numărul de segmente. p95 folosește rangul `ceil(0,95 × N)`.

## 2. Alte configurații, încărcare, memorie și recuperare

Mașina are un singur GPU integrat. Chrome și Edge au folosit același Radeon. SwiftShader este un adaptor software, iar profilul Pixel 7 rulează în Chrome desktop cu setările de mobil și încetinire CPU configurată prin CDP. Validarea pe un GPU dedicat și pe un telefon fizic rămâne deschisă. [Emularea Chromium](https://chromedevtools.github.io/devtools-protocol/tot/Emulation/#method-setCPUThrottlingRate)

| Configurație | Browser | Adaptor raportat | Probe cronometrate |
| --- | --- | --- | --- |
| Chrome · Radeon | 157.0.8103.78 | amd gcn-5 | 1.440 |
| Chrome · Radeon · 8 workeri de încărcare | 157.0.8103.78 | amd gcn-5 | 640 |
| Edge · Radeon | 157.0.4341.60 | amd gcn-5 | 480 |
| Chrome · SwiftShader | 157.0.8103.78 | google swiftshader | 360 |
| Chrome · Pixel 7 emulat și încetinire CPU ×4 | 157.0.8103.78 | amd gcn-5 | 360 |

### Comparație sub încărcare și pe rutele de compatibilitate

Cu opt workeri de încărcare și zona grea mobilă la 524.288 de elemente, p95 a fost **91,90 ms** fără indiciu și **18,80 ms** cu indiciu. Câștigurile variază cu backendul, dimensiunea și încărcarea.

| Configurație | Distribuție | Elemente | Adaptive mediană / p95 | Adaptive-work mediană / p95 | Raport mediană |
| --- | --- | --- | --- | --- | --- |
| Chrome · Radeon · 8 workeri de încărcare | Zonă grea fixă | 65.536 | 7,80 / 14,30 | 5,80 / 6,30 | 1,34× |
| Chrome · Radeon · 8 workeri de încărcare | Zonă grea fixă | 524.288 | 25,45 / 74,80 | 11,20 / 15,60 | 2,27× |
| Chrome · Radeon · 8 workeri de încărcare | Zonă grea mobilă | 65.536 | 8,35 / 14,60 | 6,15 / 7,10 | 1,36× |
| Chrome · Radeon · 8 workeri de încărcare | Zonă grea mobilă | 524.288 | 38,35 / 91,90 | 12,55 / 18,80 | 3,06× |
| Edge · Radeon | Zonă grea fixă | 65.536 | 6,45 / 41,70 | 5,60 / 7,20 | 1,15× |
| Edge · Radeon | Zonă grea fixă | 524.288 | 35,65 / 179,00 | 14,35 / 17,30 | 2,48× |
| Edge · Radeon | Zonă grea mobilă | 65.536 | 14,25 / 43,00 | 7,45 / 11,40 | 1,91× |
| Edge · Radeon | Zonă grea mobilă | 524.288 | 35,10 / 59,80 | 15,85 / 19,50 | 2,21× |
| Chrome · SwiftShader | Uniform | 65.536 | 5,75 / 6,10 | 6,10 / 6,80 | 0,94× |
| Chrome · SwiftShader | Zonă grea fixă | 65.536 | 7,80 / 14,40 | 6,10 / 6,60 | 1,28× |
| Chrome · SwiftShader | Zonă grea mobilă | 65.536 | 8,60 / 14,80 | 6,35 / 7,30 | 1,35× |
| Chrome · Pixel 7 emulat și încetinire CPU ×4 | Uniform | 65.536 | 18,15 / 28,80 | 16,40 / 42,30 | 1,11× |
| Chrome · Pixel 7 emulat și încetinire CPU ×4 | Zonă grea fixă | 65.536 | 13,35 / 28,90 | 19,55 / 47,70 | 0,68× |
| Chrome · Pixel 7 emulat și încetinire CPU ×4 | Zonă grea mobilă | 65.536 | 13,85 / 22,30 | 13,05 / 26,80 | 1,06× |

### Memorie eșantionată în izolatul JavaScript principal

Am măsurat separat câte 16 apeluri adaptive la 1.048.576 de elemente, după încălzire. Colectarea CDP a avut loc în probe de diagnostic fără workeri artificiali de încărcare. Valorile sunt maxime eșantionate și pot varia cu garbage collection. Memoria workerilor, a procesului GPU și memoria fizică totală sunt în afara acestor contoare. [Contoarele CDP](https://chromedevtools.github.io/devtools-protocol/tot/Runtime/#method-getHeapUsage)

| Profil | Eșantioane | Heap inițial / vârf | Backing storage inițial / vârf |
| --- | --- | --- | --- |
| `full` | 22 | 1,13 / 3,17 MiB | 4,13 / 51,48 MiB |
| `summary` | 22 | 1,13 / 3,65 MiB | 4,13 / 51,47 MiB |

Vârfurile de backing storage au fost aproape egale în această probă. Eliminarea înregistrărilor per segment păstrează memoria profilului independentă de numărul de segmente, iar probele de aici includ și tablourile de date, care ocupă cea mai mare parte din backing storage.

### Recuperare după pierderea GPU-ului

În patru probe am distrus un dispozitiv WebGPU real, am simulat lipsa adaptorului la prima încercare și am verificat calculul pe CPU. După expirarea intervalului de reîncercare, un apel GPU nou a refăcut dispozitivul și pipeline-ul. Medianele separate au fost 0,15 ms până la notificarea pierderii, 17,60 ms pentru fallback și 26,85 ms pentru primul calcul GPU după pauza de reîncercare. Proba reproduce o pierdere controlată, cu prima enumerare refuzată deliberat.

## Reproducere

```sh
npm ci
npm run build
npm run benchmark:estimate-work -- --channel chrome --output estimate-work-native.json
npm run benchmark:estimate-work -- --channel chrome --layouts clustered,moving --series 2 --load-workers 8 --diagnostics --output estimate-work-load.json
npm run benchmark:estimate-work -- --channel msedge --layouts clustered,moving --series 2 --warmups 6 --samples 12 --output estimate-work-edge.json
npm run benchmark:estimate-work -- --channel chrome --software --sizes 65536 --series 2 --warmups 6 --samples 12 --output estimate-work-software.json
npm run benchmark:estimate-work -- --channel chrome --mobile --sizes 65536 --series 2 --warmups 6 --samples 12 --output estimate-work-mobile.json
```

Datele brute păstrează versiunea browserului, adaptorul, configurația și hash-urile codului rulat.
[Date brute ale versiunii beta](https://github.com/IoanU/heterojs/releases/tag/v0.2.0-beta.1)
