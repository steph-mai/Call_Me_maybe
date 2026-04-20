le StateNode agit comme un "Mode Manuel" pour forcer un chemin précis parmi des choix finis (les noms des fonctions).


1. Le Nom : Un univers "Fermé" (StateNode)
Pour le nom de la fonction, on connait la liste exacte des possibilités à l'avance (fn_add, fn_greet, etc.).

Le StateNode est parfait ici : car il définit un labyrinthe dont les murs sont fixes. L'IA ne peut pas sortir des sentiers tracés. C'est une contrainte de structure totale.

2. Les Paramètres : Un univers "Ouvert" (Logit Masking)
Pour les paramètres (comme source_string ou a), c'est différent. On ne peut pas construire un Trie pour toutes les phrases possibles que l'IA pourrait vouloir écrire.

On utilise alors un filtrage dynamique : Au lieu de suivre un chemin dans un arbre, on applique des règles mathématiques sur les probabilités (les logits) au fur et à mesure :

Si le type est number : On bloque tout ce qui n'est pas un chiffre ou un point.

Si le type est string : On laisse l'IA libre, mais on surveille le guillemet de fermeture (") pour savoir quand arrêter.

Loader > Charge les données JSON.
PromptBuilder > Prépare le texte (contexte + instructions).
DecodingState > Gère les chemins autorisés (le Trie).
ConstrainedDecoder > Utilise le prompt et les chemins autorises pour générer.
Models (Pydantic) > Valide les donnees initiales et la structure finale.