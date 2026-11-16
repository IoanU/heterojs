# HeteroJS

HeteroJS este o bibliotecă TypeScript pentru calcule numerice în browser, cu workeri CPU, execuție directă pe firul apelant și WebGPU. Motorul măsoară apelurile de calcul și poate alege CPU, GPU sau o împărțire între ele. Codul și benchmarkul rulează local în browser, fără server de calcul.

[Rezultatele benchmarkurilor](docs/benchmarks.md)

## Pornire rapidă

Ai nevoie de Node.js 20.19+ din seria 20 sau de Node.js 22.12+. Din directorul proiectului rulează:

```sh
npm ci
npm run dev
```

Deschide adresa locală afișată de Vite. Pentru distribuție, `npm run build` verifică tipurile și generează pachetul ESM cu declarații în `dist/library` și pagina statică de benchmark în `dist/site`. `npm run preview` servește local rezultatul. Pagina poate fi găzduită și într-un subdirector HTTPS.

Pentru a folosi biblioteca într-o altă aplicație, descarcă pachetul versiunii de test din [GitHub Releases](https://github.com/IoanU/heterojs/releases) sau construiește-l cu `npm pack`. Dacă arhiva este lângă `package.json` în aplicația consumatoare, rulează `npm install ./heterojs-0.2.0-beta.1.tgz`.

## API TypeScript

```ts
import { HeteroJS } from 'heterojs';

const engine = await HeteroJS.create({
  strategy: 'adaptive',
  cpuWorkers: 'auto',
  gpu: 'auto',
});

try {
  const { output, profile } = await engine.compute({
    operation: 'matrixMultiply',
    inputs: [Float32Array.of(1, 2, 3, 4, 5, 6), Float32Array.of(7, 8, 9, 10, 11, 12)],
    shape: { rows: 2, inner: 3, columns: 2 },
  });
  console.log(Array.from(output)); // [58, 64, 139, 154]
  console.log(profile.strategy, profile.totalMs);
} finally {
  engine.close();
}
```

| Operație | Parametri | Rezultat |
| --- | --- | --- |
| `vectorMultiply`, `vectorAdd` | `inputs: [a, b]`, cu lungimi egale | Câte un element pentru fiecare pereche |
| `scalarMultiply` | `inputs: [a]`, `scalar` rotunjit la `f32` | Câte un element pentru fiecare valoare |
| `sum`, `min`, `max` | `inputs: [a]` | Tablou cu o singură valoare |
| `matrixMultiply` | `inputs: [a, b]`, `shape: { rows, inner, columns }` | Matrice `rows × columns`, stocată pe rânduri |

Intrările sunt tablouri `Float32Array`. Păstrează datele intacte până la finalizarea apelului. Motorul alocă un rezultat separat. La matrice, `a` conține `rows × inner` elemente și `b` conține `inner × columns`. O dimensiune interioară zero produce zerouri. Reducerile pe intrări vide întorc `0`, `+Infinity` și `-Infinity` pentru `sum`, `min` și `max`.

Reducerea `sum` și produsele matriciale folosesc precizii de acumulare diferite pe CPU și GPU. Pentru comparații numerice, alege toleranța în funcție de magnitudine și anulare. Reducerile propagă `NaN`, iar `min` și `max` păstrează zeroul cu semn. [Regulile WGSL pentru precizia în virgulă mobilă](https://www.w3.org/TR/WGSL/#floating-point-evaluation)

## Operații custom

`registerOperation` acceptă un modul CPU, o funcție WebGPU sau ambele. `compute({ operation, input })` întoarce tipul de rezultat declarat la înregistrare. Identificatorul întors de `registerOperation` aparține motorului care l-a creat. `profile.customOperation` conține numele operației.

Pentru un algoritm cu dependențe globale, de exemplu o sortare, înregistrează implementarea care primește întreaga intrare. Servește `sort.cpu.js` ca modul JavaScript accesibil workerilor.

```js
// sort.cpu.js
export function run({ values, descending }) {
  return values.toSorted((a, b) => descending ? b - a : a - b);
}
```

```ts
type SortInput = { values: number[]; descending: boolean };

const sort = engine.registerOperation<SortInput, number[]>('sort', {
  cpu: { module: new URL('./sort.cpu.js', import.meta.url) },
  estimateUnits: ({ values }) => Math.max(1, values.length),
});

const { output } = await engine.compute({
  operation: sort,
  input: { values: [3, 1, 2], descending: false },
});
console.log(output); // [1, 2, 3]
```

- Modulul CPU primește `(input, range)` și poate întoarce o valoare sau o promisiune. Motorul apelează exportul `run` implicit. Poți furniza `exportName` pentru alt export. Intrările și rezultatele de la workeri trebuie să poată fi copiate prin structured clone, inclusiv obiecte, `Map` și tablouri tipizate. O adresă relativă transmisă ca șir se rezolvă relativ la pagina apelantă. [Contractul de mesagerie al workerilor](https://html.spec.whatwg.org/multipage/workers.html#dom-worker-postmessage)
- Funcția GPU primește `(input, { device, signal, start, length, totalUnits })`. Poate folosi propriile buffere, texturi, shadere, pipeline-uri și oricâte treceri, inclusiv un algoritm cu etape dependente. Funcția trimite comenzile, citește rezultatul dacă este necesar și gestionează resursele pe care le creează. Operațiile GPU ale motorului folosesc aceeași coadă, așa că etapele GPU dintr-o funcție custom se trimit direct prin `device`. `HeteroJS.create({ gpuDevice: { requiredFeatures, requiredLimits } })` poate cere capabilități suplimentare adaptorului. [Descriptorul dispozitivului WebGPU](https://www.w3.org/TR/webgpu/#dictdef-gpudevicedescriptor)
- Adaugă `plan(input)` pentru o operație care se poate executa pe segmente independente. Planul declară `units`, opțional `maxChunkUnits`, o funcție `chunk({ start, length, totalUnits })` care pregătește datele pentru fiecare segment și `combine(parts, signal)` pentru rezultatul final. `combine` primește segmentele în ordine după `start`, fiecare cu `value` și intervalul său. Pentru `units: 0`, primește lista vidă. CPU și GPU trebuie să producă rezultate parțiale echivalente pentru orice interval permis de plan.
- Pentru rezultate mari, un `StreamingCustomPlan` poate folosi `collector.add(part, signal)` și `collector.finish(signal)`. Motorul apelează `add` pe rând, în ordinea finalizării segmentelor. O promisiune de la `add` ține ocupată unitatea de execuție a segmentului până când rezultatul este consumat. Astfel, o destinație de streaming poate controla ritmul producătorilor. `finish` rulează după toate segmentele și după un plan vid.
- Cu `cpu.transferOutput: true`, workerul întoarce `{ value, transfer: [buffer] }`, unde lista conține bufferele `ArrayBuffer` incluse în rezultat. Workerul cedează proprietatea asupra lor la publicarea rezultatului. Modul implicit copiază rezultatul prin structured clone.
- Cu `cpu.transferInput(value, range)` poți opta și pentru transferul intrărilor către workeri. Funcția primește datele segmentului CPU și întoarce bufferele `ArrayBuffer` pe care apelantul le cedează, de exemplu `(view) => [view.buffer]` pentru o copie deținută exclusiv de segment. Trimiterea detașează bufferul și toate vederile sale pe firul apelant. Folosește un buffer distinct pentru fiecare segment pe care îl transferi și păstrează datele de care au nevoie celelalte segmente. [Transferul prin structured clone](https://html.spec.whatwg.org/multipage/structured-data.html#structuredserializewithtransfer)
- `plan.estimateWork({ start, length })` declară costul relativ al unui interval. Întoarce un întreg pozitiv și aditiv pentru fiecare interval nevid. De exemplu, dacă ai cost per element, o sumă prefixată permite `prefix[start + length] - prefix[start]`. Schedulerul folosește costul pentru împărțire și dimensiunea segmentelor, apoi actualizează estimările din timpii observați. Cu `profile.estimateBasis === 'work'`, `msPerElement` este exprimat per unitate de cost estimat.
- Pentru apeluri repetate, `plan.adaptiveKey` grupează lucrări cu forme și costuri similare. `adaptive` cronometrează apeluri utile cu CPU, GPU și împărțiri fixe. Dacă există `estimateWork`, compară și planul `adaptive-chunks`. Motorul păstrează cel mult 32 de istorii per operație, separate după cheie, dimensiune, limita segmentului, cota fixă și tipul estimării. `profile.customDispatch` arată planul ales și alocarea efectivă. Planurile fără cheie folosesc schedulerul adaptiv pe segmente.

[Exemplul de histogramă ponderată](examples/weighted-histogram/index.js) împarte perechi de chei și ponderi `Uint32Array`. Modulul CPU numără ponderile direct, shaderul WebGPU adună ponderile atomic, iar `combine` însumează histogramele parțiale. Același apel funcționează cu toate cele patru strategii.

```js
import { registerWeightedHistogram } from './examples/weighted-histogram/index.js';

const histogram = registerWeightedHistogram(engine);
const { output } = await engine.compute({
  operation: histogram,
  input: { keys: Uint32Array.of(2, 0, 2), weights: Uint32Array.of(3, 4, 5), binCount: 3 },
  strategy: 'adaptive',
});
console.log(Array.from(output)); // [4, 0, 8]
```

| Strategie custom | Alegerea backendului |
| --- | --- |
| `cpu-only` / `gpu-only` | Impune backendul cerut. Operația trebuie să aibă implementarea corespunzătoare. |
| `static` | Cu un plan, `cpuShare` controlează împărțirea pe unități. Pentru un apel întreg, o cotă de cel puțin `0.5` preferă CPU. Motorul folosește implementarea disponibilă dacă backendul preferat lipsește. |
| `adaptive` | Cu un plan, schedulerul poate modifica dimensiunea segmentelor și distribuția CPU/GPU în timpul apelului. Pentru un job întreg cu două implementări, măsoară backendurile pe apeluri succesive și alege unul pe baza istoricului operației. Fiecare apel întreg se execută o singură dată. `estimateUnits` este un indiciu opțional de dimensiune pentru acest istoric și implicit este `1`. |

Profilul unui apel custom măsoară latența pe segment și timpul de coadă. Formatul generic folosește `0` pentru octeții pe care motorul nu îi poate estima și `null` pentru durata kernelului GPU. La anulare, un worker care execută un segment custom poate fi terminat și înlocuit de motor. Funcția GPU primește semnalul de anulare și rămâne responsabilă de eliberarea resurselor sale până când se încheie.

## Lanțuri de operații pe GPU

`engine.runGpuGraph(build, { signal })` înregistrează un lanț de kerneluri WGSL. `upload` încarcă o intrare, `createBuffer` rezervă spațiu pe dispozitiv, iar `dispatch` poate folosi rezultatul unui kernel anterior ca intrare. `read` înregistrează citirile finale, în ordinea din `result.outputs`. Motorul emite trecerile și copierile către bufferul de citire într-o singură trimitere WebGPU, apoi eliberează bufferele grafului.

```js
import { makeInputs, runResident } from './examples/gpu-graph/index.js';

const { a, b, expected } = makeInputs(262_144);
const { value, profiles: [profile] } = await runResident(engine, a, b);
console.log(value === expected, profile.uploadBytes, profile.readbackBytes);
```

[Exemplul complet](examples/gpu-graph/index.js) înmulțește două tablouri `Uint32Array`, aplică un prag și reduce valorile la o sumă. Kernelurile pot declara legături în mai multe grupuri, buffere `storage` și `uniform`, regiuni de buffer și constante WGSL. Dimensiunile şi offseturile pentru copii se exprimă în octeți și sunt multipli de patru. Grafurile respectă și limitele adaptorului pentru buffere, legături și număr de workgroup-uri. [Regulile WebGPU pentru copierea bufferelor](https://www.w3.org/TR/webgpu/#buffer-copy-commands)

Construiește graful sincron și păstrează intrările intacte până la finalizarea apelului. Identificatorii bufferelor au durata de viață a unui singur graf. `read(buffer, { offset, size })` citește opțional o subregiune și întoarce indexul rezultatului în `outputs`. Rezultatele sunt `ArrayBuffer` independente. Execuția folosește WebGPU explicit, iar lipsa dispozitivului produce o eroare. Anularea respinge apelul, iar lucrul deja trimis își eliberează resursele după terminare.

`profile` raportează octeții încărcați și citiți, operațiile de transfer și numărul de trimiteri explicite. Timpii de transfer măsoară apelurile de încărcare și copierea rezultatelor mapate. Timpul de sincronizare include așteptarea GPU și maparea bufferului, deci se poate suprapune cu execuția kernelurilor.

```sh
npm run build
npm run benchmark:gpu-graph -- --channel chrome --output gpu-graph-results.json
```

Runnerul verifică rezultatele față de referința CPU și compară același lanț WGSL în două moduri. Graful rezident citește doar suma finală, iar varianta în etape citește și reîncarcă cele două rezultate intermediare în JavaScript. [Rezultatele benchmarkului](docs/benchmarks.md#lanțuri-de-kerneluri-pe-gpu)

## Strategii și profilare

| Strategie | Comportament |
| --- | --- |
| `cpu-only` | Rulează în workeri CPU persistenți. Setarea `auto` folosește un worker mai puțin decât numărul de nuclee logice raportat, între 1 și 8. |
| `gpu-only` | Rulează pe WebGPU. O cerere cu lucru efectiv eșuează dacă dispozitivul lipsește sau forma nu încape în limitele sale. |
| `static` | Trimite CPU-ului primele `round(units × cpuShare)` unități. Valoarea implicită a cotei este `0.25`. Unitățile sunt elemente de vector sau rânduri de ieșire ale matricei. Revine la CPU când WebGPU lipsește sau matricea depășește limitele dispozitivului. Este strategia implicită a motorului. |
| `adaptive` | Pentru vectori și reduceri, folosește timpii segmentelor și cozile de lucru pentru a ajusta dimensiunea și backendul. Pentru matrici mari, compară latența apelurilor complete pentru CPU, GPU și un plan mixt cu un singur worker CPU. Păstrează combinația când medianele recente o favorizează clar. |

Pe CPU direct, calculul cedează firul după aproximativ 8 ms de lucru când mai rămân date. `inlineCpu: false` limitează adaptivul la workeri și WebGPU. Pentru matrici, planul mixt aliniază rândurile de pe GPU la blocurile de 8 rânduri și pornește workerul după trimiterea comenzii WebGPU.

- `getCapabilities()` raportează workerii configurați și disponibilitatea WebGPU. `gpu: 'disabled'` împiedică inițializarea adaptorului.
- `profile.requestedStrategy` este strategia cerută. `profile.strategy` arată execuția efectivă. `profile.totalMs` include planificarea, transferurile și asamblarea rezultatului.
- `profile.chunks` arată alocarea pe backend, timpii și octeții transferați. `profile.matrixDispatch` arată planul ales pentru matrici suficient de mari. `profile.fallbackReason` explică revenirea la CPU, când este disponibilă.
- `compute({ ..., profile: 'summary' })` păstrează contoarele per backend în `profile.chunkSummary` și întoarce `profile.chunks` vid. Astfel, memoria folosită pentru profil rămâne independentă de numărul segmentelor. Modul implicit, `full`, păstrează istoricul pe segmente. Timpii cumulați ai segmentelor paralele se pot suprapune.
- `getProfile()` întoarce ultimul profil finalizat cu succes. `compute()` acceptă `signal` pentru anulare. Lucrul deja trimis poate continua până se eliberează backendul. `close()` termină workerii și eliberează dispozitivul și bufferele GPU.

Când dispozitivul WebGPU se pierde, operațiile GPU active și cele din coada lui sunt respinse cu `GpuDeviceLostError` când motorul primește notificarea WebGPU. Următorul apel care cere GPU încearcă să deschidă un dispozitiv nou și să reconstruiască pipeline-urile. Dacă adaptorul lipsește, apelurile compatibile pot continua pe CPU. O inițializare eșuată este reîncercată la o cerere ulterioară după o pauză de o secundă. Operațiile custom pot folosi `device` din context ca cheie pentru un `WeakMap` cu resurse specifice dispozitivului. [Semnalarea pierderii dispozitivului în WebGPU](https://www.w3.org/TR/webgpu/#dom-gpudevice-lost)

## Benchmark reproductibil

Pagina locală permite alegerea operației, a dimensiunilor, a numărului de încălziri și a strategiei. Compară `cpu-only`, `gpu-only`, `static`, `adaptive` și referința `CPU direct`. Pentru matrici, dimensiunea din interfață este latura, iar debitul în GFLOP/s folosește convenția `2 × latură³`.

```sh
npm run build
npx playwright install chromium
npm run benchmark:browser -- --comparison all --operation matrixMultiply --sizes 257,513,768 --warmups 8 --samples 15 --series 4 --output matrix-results.json
```

Folosește `--channel chrome` sau `--channel msedge` pentru un browser instalat, respectiv `--load-workers 4` pentru încărcare CPU controlată. Pe mobil, deschide pagina găzduită, folosește aceleași setări și exportă JSON din interfață.

Mediana și p95 se calculează din apelurile cronometrate după încălzire. Fiecare serie are și primul apel pe un motor proaspăt. Comparația `vs. best CPU/GPU` împarte latența celei mai rapide dintre `cpu-only` și `gpu-only` la latența strategiei evaluate. O valoare peste `1×` indică depășirea ambelor referințe. `p95 loop delay` urmărește întârzierea unui temporizator de 4 ms ca indicator de responsivitate. Corectitudinea este verificată în afara cronometrării.

### Rezultat ilustrativ

Matrice 513 × 513, Chrome pe AMD Ryzen 7 5800H cu GPU Radeon integrat, fără workeri de încărcare, 160 de apeluri cronometrate per strategie.

| Metrică | `cpu-only` | `gpu-only` | `adaptive` |
| --- | ---: | ---: | ---: |
| Mediană | 32,3 ms | 6,2 ms | **5,7 ms** |
| p95 | 37,0 ms | 8,2 ms | **7,7 ms** |

La 768 × 768, p95 adaptiv a fost 16,3 ms, față de 15,3 ms pentru GPU. Rezultatele de performanță provin de pe un singur sistem.

### Workload real din Fractalia

Fractalia înregistrează calculul câmpului de iterații ca operație custom HeteroJS și păstrează randarea progresivă în Canvas 2D. Benchmarkul compară rendererul original cu `cpu-only`, `gpu-only`, `static` și `adaptive`. Măsoară timpul până la prima previzualizare, latența cadrului final și diferențele de imagine față de workerul original. [Rezultatele integrării experimentale](docs/benchmarks.md#fractalia)

### Estimarea costului pe segmente

`npm run benchmark:estimate-work -- --channel chrome` compară schedulerul adaptiv cu și fără `estimateWork`, pe aceleași kerneluri CPU și WebGPU. `--load-workers 8` activează încărcarea CPU. `--software` selectează SwiftShader, iar `--mobile` emulează un Pixel 7 în Chromium desktop, cu încetinire CPU. `--diagnostics` măsoară separat memoria izolatului principal și recuperarea după pierderea GPU-ului. [Metodă, rezultate și comenzi](docs/estimate-work.md)

## Verificare și cerințe de browser

```sh
npm test
npx playwright install chromium
npm run test:browser
npm run test:package
```

Testele de browser rulează pe biblioteca și pagina compilate. `HETEROJS_BROWSER_CHANNEL=chrome` selectează Chrome instalat. `HETEROJS_REQUIRE_GPU=1` cere un adaptor WebGPU. `HETEROJS_SOFTWARE_GPU=1` permite verificarea corectitudinii cu SwiftShader.

Verificarea pachetului instalează arhiva într-un proiect curat, validează declarațiile TypeScript și rulează operații prin biblioteca instalată în browser. CI execută suitele și încarcă arhiva pe Windows și Linux.

Sunt necesare module ES, Web Workers și `AbortSignal.any`. Pentru workeri încorporați într-o aplicație cu Content Security Policy, permite `blob:` în `worker-src`. WebGPU necesită HTTPS sau localhost, plus browser și adaptor compatibile. [Specificația WebGPU](https://www.w3.org/TR/webgpu/) · [Configurarea browserelor Playwright](https://playwright.dev/docs/browsers)

## Licență

Codul este distribuit sub [licența MIT](./LICENSE).
