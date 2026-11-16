# Benchmarkuri HeteroJS

Măsurătorile de mai jos provin de pe un sistem AMD Ryzen 7 5800H cu Radeon integrat. Latența variază cu dispozitivul, operația și încărcarea. [Raportul suitei `estimateWork`](estimate-work.md)

## Operații integrate

Pe Chrome, matrice 513 × 513, mediana a fost 32,3 ms cu `cpu-only`, 6,2 ms cu `gpu-only` și 5,7 ms cu `adaptive`. Valorile p95 au fost 37,0, 8,2 și 7,7 ms. La 768 × 768, GPU-ul a avut p95 de 15,3 ms, iar `adaptive` a avut 16,3 ms. Runnerul din bibliotecă reproduce comparația la dimensiunile alese.

```sh
npm run build
npm run benchmark:browser -- --channel chrome --comparison all --operation matrixMultiply --sizes 513,768
```

## Lanțuri de kerneluri pe GPU

Pentru `înmulțire → activare → sumă`, graful cu date intermediare pe GPU a obținut o accelerare mediană de 2,34–3,17× față de aceleași kerneluri cu citire și reîncărcare între etape. La 1.048.576 de elemente per intrare, medianele au fost 5,50 ms și 17,45 ms. Referința CPU a produs aceeași sumă la fiecare probă.

```sh
npm run benchmark:gpu-graph -- --channel chrome
```

## Fractalia

Integrarea experimentală a măsurat calculul câmpului de evadare pentru Mandelbrot, Julia și Burning Ship. Cu previzualizări identice, selecția adaptivă a planurilor complete a depășit fiecare backend separat după mediana ultimei treceri în 9 din 9 cazuri și după p95 în 8 din 9. S-a executat și o probă Julia pe un raster de 3840 × 2307. În acea probă, mediana calculului final a fost 179,4 ms adaptiv și 239,8 ms pe CPU. Timpul cadrului complet, cu Canvas, a fost 709,8 ms adaptiv și 688,9 ms pe CPU. [Proiectul original Fractalia](https://github.com/IoanU/fractalia)

Shaderul experimental WebGPU folosește o aritmetică diferită de cea Float64 pe CPU, iar unele orbite sensibile pot avea culori diferite. Fidelitatea imaginii se evaluează separat de latență.
