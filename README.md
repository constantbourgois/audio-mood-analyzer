# Analyseur audio (tonalité, tempo, ambiance)

Petite appli web Flask, en local, pour uploader un fichier audio (guitare,
chant, ...) et obtenir :

- **Tempo** (BPM)
- **Tonalité et mode** (majeur/mineur)
- **Ambiance / mood** (ex. "Joyeux / Énergique", "Mélancolique / Calme")
- **Carte de la chanson** : grille de mesures + moment où chaque "piste"
  (percussions, basses, accords, voix/mélodie) rentre dans le morceau
- **Morceaux similaires** dans ta propre bibliothèque locale (dossier `library/`)

## Installation

```bash
cd audio-mood-analyzer
python3 -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Formats supportés nativement : `.wav`, `.flac`, `.ogg`. Le **`.mp3` et le
`.m4a` nécessitent `ffmpeg` installé sur ta machine** (sinon l'upload de ces
formats échouera avec un message d'erreur clair). Pour les convertir en amont
si besoin : `ffmpeg -i fichier.mp3 fichier.wav`.

## Lancer l'appli

```bash
python app.py
```

Puis ouvrir http://127.0.0.1:5050 dans le navigateur.

## Utilisation

1. Choisis un fichier audio et clique sur "Analyser".
2. Coche "Ajouter à ma bibliothèque" pour les morceaux que tu veux pouvoir
   retrouver plus tard comme référence de comparaison (par exemple tes
   propres enregistrements précédents, ou des morceaux que tu apprécies).
3. Les prochaines analyses comparent automatiquement le nouveau fichier aux
   morceaux déjà présents dans `library/` et affichent les plus proches.

## Déploiement (Render)

L'appli tourne aussi en ligne, sur un vrai hébergeur (pas en local). Render a
été choisi plutôt que Vercel : Vercel fonctionne en *serverless* avec un
timeout court par requête et une limite de taille de fonction, mal adaptés
à `librosa`/`numpy`/`scipy` (un vrai morceau de quelques minutes peut prendre
20-40s à analyser). Render fait tourner un conteneur Python persistant, sans
ce genre de contrainte.

1. Sur [render.com](https://render.com), **New** → **Blueprint**, connecte ce
   dépôt GitHub. Render lit `render.yaml` et configure tout seul (build,
   start command, variable `SECRET_KEY` générée automatiquement).
2. Choisis le plan **Free** pour tester. Premier déploiement : quelques
   minutes (installation de `librosa`/`numpy`/`scipy`).
3. Une fois déployé, Render donne une URL du type
   `https://audio-mood-analyzer.onrender.com`.

**Limites du plan gratuit à connaître** :
- Le conteneur **s'endort après une période d'inactivité** ; la requête
  suivante le réveille, ce qui ajoute ~30-60s à la première analyse après
  une pause.
- Le disque est **éphémère** : tout fichier ajouté à `library/` via l'appli
  est perdu au redémarrage ou au redéploiement. Pour une bibliothèque qui
  survit dans le temps, il faudrait un disque persistant Render (payant) ou
  un stockage externe (S3, etc.) — pas fait pour l'instant, à décider si
  besoin.

Sans `render.yaml` (déploiement manuel) : Build Command
`pip install -r requirements.txt`, Start Command
`gunicorn -c gunicorn.conf.py -w 2 --timeout 120 -b 0.0.0.0:$PORT app:app`,
variable d'environnement `SECRET_KEY` à définir toi-même (une chaîne
aléatoire quelconque).

## La carte de la chanson

Décision assumée : **pas de séparation de sources** (pas de Demucs/PyTorch,
~1-2 Go de dépendances) pour rester léger à installer. Conséquence directe :
l'appli ne peut pas identifier un instrument précis (« c'est une guitare »).
Ce qu'elle repère à la place :

- **Percussions (probable)** — composante percussive du signal (séparation
  harmonique/percussive de librosa)
- **Basses (graves)** — énergie de la bande de fréquence < 250 Hz
- **Accords / harmonie (médiums)** — énergie de la bande 250-2500 Hz sur la
  composante harmonique
- **Voix / mélodie principale (probable)** — probabilité "voisée" d'un suivi
  de hauteur monophonique (pYIN)

Pour chaque piste, l'appli détecte le premier instant où elle devient
active de façon soutenue (pas juste un coup isolé), et le rattache à un
numéro de mesure. Le découpage en mesures suppose une mesure à 3 ou 4 temps
(les cas les plus courants), estimée par contraste d'accentuation — sans
jamais lire de vraie partition.

**Limite assumée à connaître** : une mélodie jouée par un instrument solo
(guitare, saxophone, synthé...) peut être détectée comme "voix" — il n'y a
pas de distinction fiable entre les deux sans séparation de sources. À
utiliser comme un guide visuel de la texture du morceau dans le temps, pas
comme une transcription instrument par instrument. Le traitement (surtout
le suivi de hauteur pour la voix) peut prendre de quelques secondes à
environ une minute selon la durée du morceau.

## Limites, en toute transparence

- **Tonalité/mode** : algorithme de Krumhansl-Schmuckler (corrélation avec
  des profils de hauteurs types), la référence classique en analyse
  musicale. Fiable sur un morceau tonal clair, moins fiable sur un passage
  très court, très dissonant ou modulant.
- **Ambiance/mood** : ce n'est **pas** un modèle entraîné sur des exemples
  réels de musique, mais une heuristique interprétable (mode majeur/mineur +
  tempo + énergie + brillance spectrale, sur le modèle valence/énergie de
  Russell). Elle donne une indication raisonnable, pas un jugement expert.
  Un vrai modèle de mood entraîné (par ex. via essentia ou un modèle audio
  pré-entraîné) donnerait de meilleurs résultats mais demande une intégration
  plus lourde.
- **Morceaux similaires** : compare uniquement aux fichiers que *toi* tu as
  ajoutés dans `library/`. Il n'y a aucune connexion à une base de données
  musicale externe (Spotify, Shazam, etc.) — en brancher une demanderait une
  clé d'API tierce et un choix de service à faire.

## Prochaines étapes possibles

- Modèle de mood entraîné (embeddings audio) au lieu de l'heuristique.
- Recherche de morceaux similaires sur un vrai catalogue externe (nécessite
  une API et une clé, à décider).
- Détection d'accords complets (pas seulement la tonalité globale).
- Séparation de sources (Demucs) pour une carte de la chanson par vrai
  instrument, et extraction des notes jouées par piste (transcription
  approximative, ex. via basic-pitch) — écarté pour l'instant pour rester
  léger à installer (~1-2 Go de dépendances en plus, traitement plus lent).
