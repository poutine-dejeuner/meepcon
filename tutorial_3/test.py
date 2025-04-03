import sys
import os


def nom_fichier():
    chemin_fichier = sys.argv[0]

    # Obtenir le nom de base du fichier (avec extension)
    nom_fichier_avec_extension = os.path.basename(chemin_fichier)

    # Obtenir le nom du fichier sans l'extension
    nom_fichier_sans_extension = os.path.splitext(nom_fichier_avec_extension)[0]

    return nom_fichier_sans_extension

n = nom_fichier()
print(n)
