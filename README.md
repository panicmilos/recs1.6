# recs1.6

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

 `train.py` se pokreće komandom `python train.py` i služi za obučavanje mreže.
 
