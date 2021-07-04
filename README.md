# ReCS1.6

ReCS1.6 predstavlja projekat koji je sposoban da prepozna i klasifikuje timove igrača iz sada već legendarne igrice Counter Strike 1.6. Igrica ima dva tima, to su "Kanteri" i "Terori" od kojih u svakom timu postoji 4 različita modela. 

## Članovi tima:

[Miloš Panić (sw19-2018)](https://github.com/panicmilos)  
[Luka Šerbedžija (sw32-2018)](https://github.com/lukaserbedzija)

## Resursi:

[Slike iz skupa podataka](https://drive.google.com/drive/u/2/folders/1JEGqHTQcQaakKBCeB5WB9aVWa_1sM1R8)   
[Anotacije slika](https://docs.google.com/spreadsheets/d/1EqSbc1H2dcpJ1exzEMBKa17dtIzj-pyN0ISjIFKpWi4/edit?usp=sharing)   
[Istrenirane težine](https://drive.google.com/drive/folders/1qV30VXTh__nHawVNBPEsrHV-v6kL4gCA?usp=sharing)    

## Korišćenje:

`main.py` se pokreće komandom `python main.py` i sadrži pomoćne funkcije za:
<ul>
  <li>Konvertovanje google sheet csv fajla(iz resursa) u csv fajl pogodan za yolo treniranje.</li>
  <li>Iscrtavanje graničnih okvira svih anotacija(iz resursa) na slike iz skupa podataka kako bi se na jednostavan način proverile anotacije skupa podataka.</li>
  <li>Funkciju za deljenje skupa podataka na dva dela sa zadatim udelom slika. Prilikom svakog deljenja je potrebno ponovo obučiti mrežu sa novim trenirajućim skupom.</li>
</ul>

`yolo/train.py` se pokreće komandom `python train.py` i služi za obučavanje mreže.
 
`yolo/detector.py` se može pokrenuti komandom `python detector.py` (postoje i opcioni parametri koji se mogu pročitati u fajlu) i u tom slučaju će uzeti sve slike iz foldera, iskoristiti istreniranu mrežu nad njima i rezultat tj pronađene granične okvire će icrtati na njima i sačuvati slike.
Ovaj fajl se takođe može pokrenuti komanom `python detector.py --aimbot true --team ct/tt` čime se u pozaditi pokreće aimbot. On uzima slike sa ekrana i iz tog razloga Counter Strike 1.6 mora biti u windowed modu. Tasteri:
<ul>
  <li>F8 služi da aktivaciju/deaktivaciju aimbota.</li>
  <li>F9 služi za promenu tima tj. da stavi aimbotu do znanja koji je trenutni tim igrača kako ne bi pucao svoje igrače.</li>
  <li>F10 služi za gašenje trajno gašenje aimbota.</li>
</ul>

`yolo/test.py` se pokreće komandom `python test.py` i služi za provlačenje testnog skupa podataka kroz istreniranu mrežu i prikupljanje metrike poput <i>accuracy</i>, <i>recall</i>, <i>precision</i>.

Po potrebi se mogu menjati PATH promenjive koje se nalaze u fajlovima sa kodom ili podacima.
