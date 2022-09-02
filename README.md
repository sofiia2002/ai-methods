# ai-methods

## Wstęp

W ramach danego repozytorium znajdują się rozwiązania określonych problemów z dizedziny data science, są tutaj rozwiązania zaprojektowane na podstawie penych metodyk z dziedziny uczenia maszynowego, przeszukiwania i optymalizacji, deep learning oraz dziedzin pokrewnych. 

Wszystkie zaproponowane rozwiązania zostały zaimplementowane samodzielnie (prócz rozwiązania dotyczącego sieci neuronowych), przy pomocy ogólnodostępnych materiałów oraz kilku książek dotyczących metod wykorzystywanych w ramach metod sztucznej inteligencji. Wszystkie rozwiązania są zaimplementowane w języku **Python** oraz przy wykorzystaniu dodatkowych bibliotek, takich jak **numpy**, **matplotlib**, **seaborn**, **pandas** i **scikt-learn**.

## Uporządkowanie rozwiązań

Każdy z katalogów znajdujących się w repozytorium zawiera jedną lub kilka metod z opisanej dziedziny algorytmów sztucznej inteligencji, mając na celu rozwiązać konkretny problem oraz zbadać wpływ wykorzystywanych hipermaparametrów w każdym z przypadków. 

Opis rozwiązaywanych problemów w przypadku każdej z metod:

- **ann**: implementacja perceptronu wielowarstwowego oraz wybranego algorytmu optymalizacji gradientowej (najszybszego spadku) z algorytmem propagacji wstecznej, przetestowanie perceptronu w ramach kilku problemów klasyfikacji, a następnie wytrenowania go do klasyfikacji zbioru danych MNIST.
- **evolution**
  - *genetic-algorithm*: implementacja algorytmu genetycznego z selekcją ruletkową krzyżowaniem jednopunktowym oraz sukcjesją generacyjną, implementacja następnie została wykorzystana do znalezienia maksymalnego zysku dla problemu opisanego w środku notebook'u, a następnie został zbadany wpływ hiperparamtrów na otrzymywane rozwiązanie (oraz znalezienie najlepszego takiego zestawu).
- **machine-learning**
  - *bayes-classificator*: implementacja naiwnego klasyfikatora Bayesa, wykorzystanie stworzonego algorytmu do stworzenia klasyfikatora dla zbioru danych iris, gdzie do zbadania jakości został wykorzystany algorytm n-krotmej walidacji krzyżowej.
  - *classification*: implementacja drzewa decyzyjnego tworzonego algorytmem ID3 z ograniczeniem maksymalnej głębokości drzewa, a następnie wykorzystanie stworzonego algorytmu do stworzenia i zbadania jakości klasyfikatorów dla zbioru danych breast cancer.
  - *q-learning*: implementacja algorytmu Q-learning, a następnie stworzenie agentu rozwiązującego problem Taxi (dostępnego w pakiecie gym).
- **optimalization**
  - *gradient-descent*: implementacja algorytmu gradientu prostego oraz zastosowanie go do znalezienia minimum wybranych funkjci oraz zbadanie wpływu rozmiaru kroku dla różnych punktów początkowych.
- **search**
  - *minmax-alpha-beta-pruning*: implementacja algorytmu min-max z obcinaniem alfa beta, gdzie dla różnych ruchów o tej samej jakości algorytm zwraca losowy z nich, a następnie wykorzystanie implementacji do porównania jakości dla różnych głębokości przeszukiwania dla gry ConnectFour.
  
## Dodatkowo

Podziękowania [Jakub Łyskawa](https://github.com/lychanl) za ukształtowanie zadań i sprawdzenie poprawności rozwiązań w ramach przedmiotu WSi realizowanego przez WUT.
