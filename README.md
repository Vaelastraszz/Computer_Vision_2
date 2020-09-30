# Computer_Vision_2

Ci joint l'ensemble des fichiers requis pour arriver à creer une class activation map à partir d'une poto et du modèle déja entrainé

Fichiers joints:
- Le notebook sur lequel j'ai testé cette méthode
- Le script qui permet de réaliser la génération d'une CAM
- L'image sur laquelle j'ai testé et la cam d'activation liée

# Process pour créer la CAM

- Récupération du modèle entraîné
- Récupération de la sortie du dernier bloc de convolution / des poids du premier FC layer et second FC layer
- Multiplication de la sortie du bloc de conv par les poids du premier layer mat (10,10,2048)*(2048,1024) puis de ce dernier produit par les poids du second layer pour avoir une image de type (10,10,1)
- On resize de la taille de l'image en input 299*299
- Puis on superpose la CAM et l'image de base 

# Crédit

- J'ai utilisé des blogs sur la CAM de medium notamment pour me donner des intuitions sur comment coder cette tâche

# Comment exécuter le code

- Télécharger le script 
- Avoir les librairies installées et le modèle téléchargé, changer le chemin de l'image pour tester sur une autre image
- Exécuter le script
