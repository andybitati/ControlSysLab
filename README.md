# ControlSysLab

**Application desktop complète pour l'analyse et la conception de systèmes de contrôle**

ControlSysLab est une application Python/PyQt5 offrant une interface intuitive pour analyser et concevoir des lois de commande pour systèmes dynamiques linéaires et non-linéaires, avec visualisation en temps réel et simulation avancée.

## 🚀 Fonctionnalités Principales

### 1. **Analyse d'État**
- Analyse de systèmes à partir de l'équation d'état (matrices A, B, C, D)
- Calcul automatique des pôles et zéros
- Réponses temporelles (impulsionnelle, indicielle, sinusoïdale)
- Diagrammes de Bode et analyse fréquentielle
- Évaluation de stabilité automatique

### 2. **Contrôlabilité et Observabilité**
- Calcul des matrices de contrôlabilité et observabilité
- Vérification automatique avec verdicts explicites
- Analyse des rangs et détection des états non accessibles
- Visualisations des valeurs singulières
- Décomposition de Kalman

### 3. **Commande par Rétroaction d'État**
- Placement de pôles par spécifications temporelles ou pôles directs
- Conception automatique du gain K
- Simulation en boucle fermée avec consigne et perturbations
- Visualisation de l'évolution des états et signaux de commande
- Analyse de performance complète

### 4. **Commande par Rétroaction de Sortie**
- Conception d'observateurs de Luenberger
- Système complet contrôleur + observateur
- Comparaison boucle complète vs partielle
- Analyse de l'erreur d'estimation
- Configuration flexible des mesures partielles

### 5. **Systèmes Non-Linéaires**
- Éditeur sécurisé d'équations différentielles Python
- Linéarisation automatique par calcul jacobien
- Méthode de Lyapunov avec fonctions quadratiques
- Portraits de phase interactifs
- Exemples prédéfinis (pendule, Van der Pol, Lorenz, etc.)

### 6. **Régulateurs PID**
- Méthodes automatiques : Ziegler-Nichols, Cohen-Coon
- Réglage manuel avec sliders interactifs
- Anti-windup configurable
- Comparaison graphique des méthodes
- Analyse fréquentielle du système bouclé

## 🎨 Interface Utilisateur

- **Design moderne** avec palette 3 couleurs : Blanc ciel (#F0F8FF), Bleu (#0D47A1), Rouge (#D32F2F)
- **Navigation intuitive** avec sidebar et en-tête
- **Éditeur de matrices amélioré** avec cellules larges et confortables
- **Visualisation temps réel** avec matplotlib intégré
- **Actions rapides** : Nouveau, Ouvrir, Enregistrer, Exporter, Capturer

## 📋 Prérequis Techniques

- **Python 3.10+**
- **PyQt5** - Interface graphique
- **NumPy** - Calculs numériques  
- **SciPy** - Algorithmes scientifiques
- **python-control** - Systèmes de contrôle
- **matplotlib** - Graphiques (backend Qt5Agg)
- **sympy** - Calcul symbolique
- **pandas** - Manipulation de données
- **reportlab** - Génération PDF

## 🛠️ Installation

### Méthode recommandée

```bash
# Cloner le projet
git clone https://github.com/votre-repo/controlsyslab.git
cd controlsyslab

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python -m controlsyslab
