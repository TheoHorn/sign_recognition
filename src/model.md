# Modèle de reconnaissance de l'alphabet en langue des signes
## Initialisation du modèle
On commence par importer les librairies nécessaires pour la création du modèle. Puis on importe les données d'entraînement et de validation.
Ensuite, on convertit le tout en tableau numpy pour pouvoir les utiliser dans le modèle. On sépare les données entre les informations utiles et les labels correspondants à la lettre de l'alphabet. On normalise ensuite les données entre 0 et 1 pour faciliter l'entraînement du modèle. Suite à cela, on redimensionne les données pour qu'elles soient compatibles avec le modèle, ici 28x28 pixels. Finalement, on convertit les labels en catégories pour pouvoir les utiliser dans le modèle.

## Réseau de neurones convolutif (CNN) composé de 8 couches

### Première couche : 
Cette couche effectue une opération de convolution sur les données d'entrée (de forme 28x28) avec 32 filtres de 3x3 et une fonction d'activation ReLU. Les filtres sont des matrices de poids que le CNN apprend pendant l'entraînement pour détecter des caractéristiques locales dans les images. La fonction d'activation ReLU (Rectified Linear Unit) est utilisée pour introduire une non-linéarité dans le modèle et aider à prévenir le problème de saturation des gradients. La forme de sortie est (28, 28, 32), car nous avons maintenant 32 caractéristiques pour chaque pixel de l'image.

### Deuxième couche : 
Cette couche effectue une opération de pooling max sur les données d'entrée avec une taille de fenêtre de 2x2. Le pooling max est une opération de sous-échantillonnage qui réduit la dimensionnalité des données d'entrée en sélectionnant la valeur maximale dans chaque fenêtre de 2x2. Cela aide à réduire la surcharge de calcul et à prévenir le surajustement en forçant le CNN à détecter des caractéristiques plus robustes et plus généralisables. La forme de sortie est (14, 14, 32), car nous avons réduit de moitié la hauteur et la largeur de l'image.

### Troisième couche : 
Cette couche effectue une autre opération de convolution sur les données d'entrée avec 64 filtres de 3x3 et une fonction d'activation ReLU. Cette couche détecte des caractéristiques plus complexes et plus abstraites dans les images en combinant les caractéristiques locales détectées par la première couche de convolution. La forme de sortie est (14, 14, 64), car nous avons maintenant 64 caractéristiques pour chaque pixel de l'image.

### Quatrième couche : 
Cette couche effectue une autre opération de pooling max sur les données d'entrée avec une taille de fenêtre de 2x2. Cette couche réduit encore la dimensionnalité des données d'entrée et aide à prévenir le surajustement. La forme de sortie est (7, 7, 64), car nous avons réduit de moitié la hauteur et la largeur de l'image.

### Cinquième couche :
Cette couche aplatit les données d'entrée en un vecteur unidimensionnel. Cette couche est nécessaire pour connecter les caractéristiques détectées par les couches de convolution et de pooling aux couches entièrement connectées suivantes. La forme de sortie est (3136,), car nous avons aplati le tensor de 7x7x64 en un vecteur de 3136 éléments.

### Sixième couche : 
Cette couche est une couche entièrement connectée avec 64 nœuds et une fonction d'activation ReLU. Cette couche combine les caractéristiques détectées par les couches de convolution et de pooling en une représentation globale de l'image. La fonction d'activation ReLU est utilisée pour introduire une non-linéarité dans le modèle.

### Septième couche : 
Cette couche est une autre couche entièrement connectée avec 26 nœuds (car meme si le J et le Z ne peuvent pas être représenté, on a A=1, ..., Y=25, Z=26) et une fonction d'activation softmax. Cette couche produit une distribution de probabilité sur les 26 lettres. La fonction d'activation softmax est utilisée pour normaliser les sorties de la couche en une distribution de probabilité valide.

## Compilation du modèle :
- optimizer='adam' : L'optimiseur Adam est un algorithme d'optimisation de premier ordre qui est couramment utilisé pour l'entraînement de modèles d'apprentissage profond. Il combine les avantages des algorithmes d'optimisation de gradient stochastique avec momentum et de l'adaptation du taux d'apprentissage.
- loss='sparse_categorical_crossentropy' : La fonction de perte de croix entropique catégorielle est couramment utilisée pour les tâches de classification multiclasse. Dans ce cas, nous utilisons la variante "sparse" de la fonction de perte, car nos étiquettes sont des entiers (et non des vecteurs one-hot).
- metrics=['accuracy'] : Nous souhaitons suivre la précision du modèle pendant l'entraînement. La précision est le pourcentage d'images correctement classées par le modèle.

## Entraînement du modèle :
Les données d'entrainement sont composés de 26 classes et de 27456 images. On a utilisé 20% des données pour la validation. Le modèle est entraîné sur 10 époques (c'est à dire que l'on va passer 10 fois sur l'ensemble des données d'entraînement) avec un batch_size de 32 (c'est à dire que l'on va mettre à jour les poids du modèle après avoir vu 32 images).

## Évaluation du modèle :
Après l'entraînement, on évalue le modèle sur les données de validation pour voir comment il se comporte sur des données qu'il n'a pas vues pendant l'entraînement. On obtient une précision de 0.91, ce qui signifie que le modèle est capable de prédire correctement la lettre de l'alphabet dans 91% des cas.
