#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet Contrôlabilité et Observabilité
Analyse de la contrôlabilité et observabilité des systèmes linéaires
"""

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTextEdit, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal

# ✅ Corrigé : imports relatifs
from ..widgets_common import MatrixEditor, PlotCanvas
from ...core.ctrb_obsv import ctrb_matrix, obsv_matrix, is_controllable, is_observable
from ...core.utils import validate_system_matrices

class TabCtrbObsv(QWidget):
    """Onglet d'analyse de contrôlabilité et observabilité"""
    
    system_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Zone de gauche : paramètres et contrôles
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Éditeur de matrices
        self.matrix_editor = MatrixEditor("Système à Analyser")
        left_layout.addWidget(self.matrix_editor)
        
        # Boutons d'analyse
        analysis_group = QGroupBox("Analyse")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.btn_analyze_ctrb = QPushButton("Analyser Contrôlabilité")
        self.btn_analyze_ctrb.setObjectName("actionButton")
        self.btn_analyze_ctrb.clicked.connect(self.analyze_controllability)
        analysis_layout.addWidget(self.btn_analyze_ctrb)
        
        self.btn_analyze_obsv = QPushButton("Analyser Observabilité")
        self.btn_analyze_obsv.setObjectName("actionButton")
        self.btn_analyze_obsv.clicked.connect(self.analyze_observability)
        analysis_layout.addWidget(self.btn_analyze_obsv)
        
        self.btn_analyze_both = QPushButton("Analyse Complète")
        self.btn_analyze_both.setObjectName("validateButton")
        self.btn_analyze_both.clicked.connect(self.analyze_complete)
        analysis_layout.addWidget(self.btn_analyze_both)
        
        left_layout.addWidget(analysis_group)
        
        # Zone de résultats textuels
        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        left_layout.addWidget(self.results_text)
        
        main_splitter.addWidget(left_widget)
        
        # Zone de droite : visualisations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Canvas pour les visualisations
        self.plot_canvas = PlotCanvas("Matrices de Contrôlabilité/Observabilité")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de visualisation
        viz_buttons_layout = QHBoxLayout()
        
        viz_buttons = [
            ("Matrice Wc", self.plot_controllability_matrix),
            ("Matrice Wo", self.plot_observability_matrix),
            ("Valeurs Singulières", self.plot_singular_values),
            ("États Accessibles", self.plot_accessible_states)
        ]
        
        for text, callback in viz_buttons:
            btn = QPushButton(text)
            btn.setObjectName("plotButton")
            btn.clicked.connect(callback)
            viz_buttons_layout.addWidget(btn)
        
        right_layout.addLayout(viz_buttons_layout)
        main_splitter.addWidget(right_widget)
        
        # Proportions du splitter
        main_splitter.setSizes([400, 800])
        
    def setup_connections(self):
        """Configuration des connexions"""
        self.matrix_editor.matrix_changed.connect(self.on_matrices_changed)
    
    def on_matrices_changed(self, matrices: Dict[str, np.ndarray]):
        """Gestion du changement de matrices"""
        self.current_system.update(matrices)
        self.clear_results()
    
    def clear_results(self):
        """Effacer les résultats"""
        self.results_text.clear()
        self.plot_canvas.clear_plot()
    
    def analyze_controllability(self):
        """Analyser la contrôlabilité"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et B requises pour l'analyse de contrôlabilité")
            return
        
        try:
            A, B = matrices['A'], matrices['B']
            
            # Calcul de la matrice de contrôlabilité
            Wc = ctrb_matrix(A, B)
            rank_Wc = np.linalg.matrix_rank(Wc)
            n = A.shape[0]
            
            # Test de contrôlabilité
            controllable = is_controllable(A, B)
            
            # Valeurs singulières
            singular_values = np.linalg.svd(Wc, compute_uv=False)
            
            # Affichage des résultats
            results = f"""
ANALYSE DE CONTRÔLABILITÉ

Dimensions du système:
- Nombre d'états (n): {n}
- Nombre d'entrées (m): {B.shape[1]}

Matrice de contrôlabilité Wc:
Dimensions: {Wc.shape[0]}×{Wc.shape[1]}
Rang: {rank_Wc}
Rang requis: {n}

Verdict: {'CONTRÔLABLE' if controllable else 'NON CONTRÔLABLE'}

Valeurs singulières de Wc:
{self.format_array(singular_values)}

Conditionnement: {np.max(singular_values) / np.min(singular_values):.2e}

Interprétation:
{self.get_controllability_interpretation(controllable, rank_Wc, n)}
            """
            
            self.results_text.setText(results.strip())
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'analyse de contrôlabilité: {str(e)}")
    
    def analyze_observability(self):
        """Analyser l'observabilité"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et C requises pour l'analyse d'observabilité")
            return
        
        try:
            A, C = matrices['A'], matrices['C']
            
            # Calcul de la matrice d'observabilité
            Wo = obsv_matrix(A, C)
            rank_Wo = np.linalg.matrix_rank(Wo)
            n = A.shape[0]
            
            # Test d'observabilité
            observable = is_observable(A, C)
            
            # Valeurs singulières
            singular_values = np.linalg.svd(Wo, compute_uv=False)
            
            # Affichage des résultats
            results = f"""
ANALYSE D'OBSERVABILITÉ

Dimensions du système:
- Nombre d'états (n): {n}
- Nombre de sorties (p): {C.shape[0]}

Matrice d'observabilité Wo:
Dimensions: {Wo.shape[0]}×{Wo.shape[1]}
Rang: {rank_Wo}
Rang requis: {n}

Verdict: {'OBSERVABLE' if observable else 'NON OBSERVABLE'}

Valeurs singulières de Wo:
{self.format_array(singular_values)}

Conditionnement: {np.max(singular_values) / np.min(singular_values):.2e}

Interprétation:
{self.get_observability_interpretation(observable, rank_Wo, n)}
            """
            
            self.results_text.setText(results.strip())
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'analyse d'observabilité: {str(e)}")
    
    def analyze_complete(self):
        """Analyse complète contrôlabilité + observabilité"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A, B et C requises pour l'analyse complète")
            return
        
        try:
            A, B, C = matrices['A'], matrices['B'], matrices['C']
            
            # Analyses séparées
            Wc = ctrb_matrix(A, B)
            Wo = obsv_matrix(A, C)
            
            rank_Wc = np.linalg.matrix_rank(Wc)
            rank_Wo = np.linalg.matrix_rank(Wo)
            n = A.shape[0]
            
            controllable = is_controllable(A, B)
            observable = is_observable(A, C)
            
            # Grammiens de contrôlabilité et observabilité
            Gc = self.compute_controllability_gramian(A, B)
            Go = self.compute_observability_gramian(A, C)
            
            # Affichage des résultats complets
            results = f"""
ANALYSE COMPLÈTE DU SYSTÈME

=== PROPRIÉTÉS GÉNÉRALES ===
Dimensions: {n} états, {B.shape[1]} entrées, {C.shape[0]} sorties

=== CONTRÔLABILITÉ ===
Matrice Wc: {Wc.shape[0]}×{Wc.shape[1]}, Rang: {rank_Wc}/{n}
Verdict: {'CONTRÔLABLE' if controllable else 'NON CONTRÔLABLE'}

=== OBSERVABILITÉ ===
Matrice Wo: {Wo.shape[0]}×{Wo.shape[1]}, Rang: {rank_Wo}/{n}
Verdict: {'OBSERVABLE' if observable else 'NON OBSERVABLE'}

=== SYNTHÈSE ===
{self.get_complete_synthesis(controllable, observable)}

=== RECOMMANDATIONS ===
{self.get_recommendations(controllable, observable, rank_Wc, rank_Wo, n)}
            """
            
            self.results_text.setText(results.strip())
            
            # Visualisation automatique
            self.plot_controllability_matrix()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'analyse complète: {str(e)}")
    
    def compute_controllability_gramian(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calcul du grammien de contrôlabilité"""
        try:
            from scipy.linalg import solve_lyapunov
            # Résolution de A*Gc + Gc*A.T + B*B.T = 0
            return solve_lyapunov(A, -B @ B.T)
        except:
            # Approximation par intégration numérique
            n = A.shape[0]
            dt = 0.01
            t_final = 10.0
            t = np.arange(0, t_final, dt)
            
            Gc = np.zeros((n, n))
            for ti in t:
                eAt = self.matrix_exponential(A * ti)
                Gc += eAt @ B @ B.T @ eAt.T * dt
            
            return Gc
    
    def compute_observability_gramian(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Calcul du grammien d'observabilité"""
        try:
            from scipy.linalg import solve_lyapunov
            # Résolution de A.T*Go + Go*A + C.T*C = 0
            return solve_lyapunov(A.T, -C.T @ C)
        except:
            # Approximation par intégration numérique
            n = A.shape[0]
            dt = 0.01
            t_final = 10.0
            t = np.arange(0, t_final, dt)
            
            Go = np.zeros((n, n))
            for ti in t:
                eAt = self.matrix_exponential(A.T * ti)
                Go += eAt @ C.T @ C @ eAt.T * dt
            
            return Go
    
    def matrix_exponential(self, A: np.ndarray) -> np.ndarray:
        """Calcul de l'exponentielle matricielle"""
        from scipy.linalg import expm
        return expm(A)
    
    def format_array(self, arr: np.ndarray) -> str:
        """Formatage d'un tableau numpy"""
        return '\n'.join([f"  {val:.6f}" for val in arr])
    
    def get_controllability_interpretation(self, controllable: bool, rank: int, n: int) -> str:
        """Interprétation de la contrôlabilité"""
        if controllable:
            return "Le système est complètement contrôlable. Tous les états peuvent être atteints."
        else:
            uncontrollable_states = n - rank
            return f"Le système n'est pas complètement contrôlable. {uncontrollable_states} état(s) ne peuvent pas être contrôlés."
    
    def get_observability_interpretation(self, observable: bool, rank: int, n: int) -> str:
        """Interprétation de l'observabilité"""
        if observable:
            return "Le système est complètement observable. Tous les états peuvent être déterminés."
        else:
            unobservable_states = n - rank
            return f"Le système n'est pas complètement observable. {unobservable_states} état(s) ne peuvent pas être observés."
    
    def get_complete_synthesis(self, controllable: bool, observable: bool) -> str:
        """Synthèse complète des propriétés"""
        if controllable and observable:
            return "Système CONTRÔLABLE et OBSERVABLE → Conception optimale possible"
        elif controllable and not observable:
            return "Système CONTRÔLABLE mais NON OBSERVABLE → Observateur requis"
        elif not controllable and observable:
            return "Système NON CONTRÔLABLE mais OBSERVABLE → Contrôle limité possible"
        else:
            return "Système NON CONTRÔLABLE et NON OBSERVABLE → Restructuration nécessaire"
    
    def get_recommendations(self, controllable: bool, observable: bool, 
                          rank_c: int, rank_o: int, n: int) -> str:
        """Recommandations basées sur l'analyse"""
        recommendations = []
        
        if not controllable:
            recommendations.append(f"• Ajouter des actionneurs pour améliorer la contrôlabilité")
            recommendations.append(f"• {n - rank_c} mode(s) non contrôlable(s) détecté(s)")
        
        if not observable:
            recommendations.append(f"• Ajouter des capteurs pour améliorer l'observabilité")
            recommendations.append(f"• {n - rank_o} mode(s) non observable(s) détecté(s)")
        
        if controllable and observable:
            recommendations.append("• Système optimal pour la conception de contrôleurs")
            recommendations.append("• Placement de pôles et observateurs possibles")
        
        return '\n'.join(recommendations) if recommendations else "Aucune recommandation spécifique"
    
    def plot_controllability_matrix(self):
        """Visualiser la matrice de contrôlabilité"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et B requises")
            return
        
        try:
            A, B = matrices['A'], matrices['B']
            Wc = ctrb_matrix(A, B)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Visualisation de la matrice
            im = ax.imshow(np.abs(Wc), cmap='Blues', interpolation='nearest')
            ax.set_title('Matrice de Contrôlabilité |Wc|')
            ax.set_xlabel('Colonnes')
            ax.set_ylabel('Lignes')
            
            # Colorbar
            cbar = self.plot_canvas.figure.colorbar(im, ax=ax)
            cbar.set_label('Magnitude')
            
            # Annotations
            for i in range(Wc.shape[0]):
                for j in range(Wc.shape[1]):
                    text = ax.text(j, i, f'{Wc[i, j]:.2f}',
                                 ha="center", va="center", color="red", fontsize=8)
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de visualisation: {str(e)}")
    
    def plot_observability_matrix(self):
        """Visualiser la matrice d'observabilité"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et C requises")
            return
        
        try:
            A, C = matrices['A'], matrices['C']
            Wo = obsv_matrix(A, C)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Visualisation de la matrice
            im = ax.imshow(np.abs(Wo), cmap='Reds', interpolation='nearest')
            ax.set_title('Matrice d\'Observabilité |Wo|')
            ax.set_xlabel('Colonnes')
            ax.set_ylabel('Lignes')
            
            # Colorbar
            cbar = self.plot_canvas.figure.colorbar(im, ax=ax)
            cbar.set_label('Magnitude')
            
            # Annotations
            for i in range(min(Wo.shape[0], 10)):  # Limiter pour la lisibilité
                for j in range(Wo.shape[1]):
                    text = ax.text(j, i, f'{Wo[i, j]:.2f}',
                                 ha="center", va="center", color="blue", fontsize=8)
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de visualisation: {str(e)}")
    
    def plot_singular_values(self):
        """Visualiser les valeurs singulières"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A, B et C requises")
            return
        
        try:
            A, B, C = matrices['A'], matrices['B'], matrices['C']
            
            Wc = ctrb_matrix(A, B)
            Wo = obsv_matrix(A, C)
            
            sv_c = np.linalg.svd(Wc, compute_uv=False)
            sv_o = np.linalg.svd(Wo, compute_uv=False)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            indices_c = range(1, len(sv_c) + 1)
            indices_o = range(1, len(sv_o) + 1)
            
            ax.semilogy(indices_c, sv_c, 'bo-', linewidth=2, label='Contrôlabilité')
            ax.semilogy(indices_o, sv_o, 'ro-', linewidth=2, label='Observabilité')
            
            ax.set_xlabel('Index')
            ax.set_ylabel('Valeurs Singulières (log)')
            ax.set_title('Valeurs Singulières des Matrices Wc et Wo')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de visualisation: {str(e)}")
    
    def plot_accessible_states(self):
        """Visualiser les états accessibles (analyse modale)"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A, B et C requises")
            return
        
        try:
            A = matrices['A']
            
            # Calcul des modes (valeurs propres et vecteurs propres)
            eigenvals, eigenvecs = np.linalg.eig(A)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Plan complexe avec modes
            ax.scatter(np.real(eigenvals), np.imag(eigenvals), 
                      s=100, c='blue', marker='x', linewidth=3, label='Modes du système')
            
            # Cercle unité pour référence de stabilité
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Cercle unité')
            
            # Axe de stabilité (Re = 0)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Axe de stabilité')
            
            ax.set_xlabel('Partie Réelle')
            ax.set_ylabel('Partie Imaginaire')
            ax.set_title('Modes du Système (États Propres)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Annotations des modes
            for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
                ax.annotate(f'Mode {i+1}', 
                           (np.real(val), np.imag(val)),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red')
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de visualisation: {str(e)}")
    
    def update_system(self, system_data: Dict[str, Any]):
        """Mise à jour du système depuis l'extérieur"""
        self.current_system.update(system_data)
        if 'A' in system_data:
            self.matrix_editor.set_matrices(system_data)
    
    def export_data(self):
        """Exporter les données de contrôlabilité/observabilité"""
        # TODO: Implémenter l'export
        QMessageBox.information(self, "Export", "Export des données d'analyse à implémenter")
    
    def capture_figure(self):
        """Capturer la figure actuelle"""
        self.plot_canvas.save_figure()
