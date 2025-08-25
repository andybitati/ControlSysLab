#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet Commande par Rétroaction de Sortie
Conception d'observateurs de Luenberger et commande par rétroaction de sortie
"""

import numpy as np
from typing import Dict, Any, List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTextEdit, QMessageBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..widgets_common import MatrixEditor, PlotCanvas, ParameterPanel
from ...core.place_design import design_observer, design_state_feedback
from ...core.lti_tools import time_response
from ...core.ctrb_obsv import is_observable, is_controllable
from ...core.utils import validate_system_matrices


class TabOutputFeedback(QWidget):
    """Onglet de conception de commande par rétroaction de sortie avec observateur"""
    
    system_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.designed_K = None
        self.designed_L = None
        self.observer_poles = []
        self.controller_poles = []
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
        
        # Zone de gauche : paramètres et conception
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Éditeur de matrices
        self.matrix_editor = MatrixEditor("Système à Contrôler")
        left_layout.addWidget(self.matrix_editor)
        
        # Configuration de l'observateur
        self.setup_observer_config(left_layout)
        
        # Configuration du contrôleur
        self.setup_controller_config(left_layout)
        
        # Boutons de conception
        design_group = QGroupBox("Conception")
        design_layout = QVBoxLayout(design_group)
        
        self.btn_verify = QPushButton("Vérifier Propriétés Système")
        self.btn_verify.setObjectName("actionButton")
        self.btn_verify.clicked.connect(self.verify_system_properties)
        design_layout.addWidget(self.btn_verify)
        
        self.btn_design_observer = QPushButton("Concevoir Observateur")
        self.btn_design_observer.setObjectName("actionButton")
        self.btn_design_observer.clicked.connect(self.design_observer)
        design_layout.addWidget(self.btn_design_observer)
        
        self.btn_design_controller = QPushButton("Concevoir Contrôleur")
        self.btn_design_controller.setObjectName("actionButton")
        self.btn_design_controller.clicked.connect(self.design_controller)
        design_layout.addWidget(self.btn_design_controller)
        
        self.btn_simulate = QPushButton("Simuler Système Complet")
        self.btn_simulate.setObjectName("validateButton")
        self.btn_simulate.clicked.connect(self.simulate_complete_system)
        design_layout.addWidget(self.btn_simulate)
        
        left_layout.addWidget(design_group)
        
        # Zone de résultats textuels
        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        left_layout.addWidget(self.results_text)
        
        main_splitter.addWidget(left_widget)
        
        # Zone de droite : visualisations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Canvas pour les graphiques
        self.plot_canvas = PlotCanvas("Commande par Rétroaction de Sortie")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de visualisation
        viz_buttons_layout = QHBoxLayout()
        
        viz_buttons = [
            ("Réponse Complète", self.plot_complete_response),
            ("Erreur d'Estimation", self.plot_estimation_error),
            ("États vs Estimés", self.plot_states_comparison),
            ("Performance Contrôleur", self.plot_controller_performance)
        ]
        
        for text, callback in viz_buttons:
            btn = QPushButton(text)
            btn.setObjectName("plotButton")
            btn.clicked.connect(callback)
            viz_buttons_layout.addWidget(btn)
        
        right_layout.addLayout(viz_buttons_layout)
        main_splitter.addWidget(right_widget)
        
        # Proportions du splitter
        main_splitter.setSizes([500, 700])
        
    def setup_observer_config(self, parent_layout):
        """Configuration de l'observateur"""
        observer_group = QGroupBox("Configuration de l'Observateur")
        observer_layout = QVBoxLayout(observer_group)
        
        # Mesures disponibles
        measures_layout = QHBoxLayout()
        measures_layout.addWidget(QLabel("Sorties mesurées:"))
        
        self.combo_measured_outputs = QComboBox()
        self.combo_measured_outputs.addItems([
            "Toutes les sorties",
            "Sortie partielle"
        ])
        measures_layout.addWidget(self.combo_measured_outputs)
        
        measures_layout.addStretch()
        observer_layout.addLayout(measures_layout)
        
        # Spécifications des pôles de l'observateur
        poles_layout = QHBoxLayout()
        poles_layout.addWidget(QLabel("Vitesse observateur (facteur):"))
        
        self.spin_observer_speed = QDoubleSpinBox()
        self.spin_observer_speed.setRange(2.0, 20.0)
        self.spin_observer_speed.setValue(5.0)
        self.spin_observer_speed.setDecimals(1)
        self.spin_observer_speed.setSuffix("× plus rapide")
        poles_layout.addWidget(self.spin_observer_speed)
        
        poles_layout.addStretch()
        observer_layout.addLayout(poles_layout)
        
        # Bruit de mesure
        noise_layout = QHBoxLayout()
        
        self.checkbox_add_noise = QCheckBox("Ajouter bruit de mesure")
        noise_layout.addWidget(self.checkbox_add_noise)
        
        self.spin_noise_level = QDoubleSpinBox()
        self.spin_noise_level.setRange(0.0, 1.0)
        self.spin_noise_level.setValue(0.01)
        self.spin_noise_level.setDecimals(3)
        self.spin_noise_level.setSuffix(" %")
        self.spin_noise_level.setEnabled(False)
        noise_layout.addWidget(self.spin_noise_level)
        
        self.checkbox_add_noise.toggled.connect(self.spin_noise_level.setEnabled)
        
        noise_layout.addStretch()
        observer_layout.addLayout(noise_layout)
        
        parent_layout.addWidget(observer_group)
        
    def setup_controller_config(self, parent_layout):
        """Configuration du contrôleur"""
        controller_group = QGroupBox("Configuration du Contrôleur")
        controller_layout = QVBoxLayout(controller_group)
        
        # Type de contrôleur
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type de contrôleur:"))
        
        self.combo_controller_type = QComboBox()
        self.combo_controller_type.addItems([
            "Rétroaction d'état estimé",
            "Régulateur PI basé observation",
            "Contrôleur robuste"
        ])
        type_layout.addWidget(self.combo_controller_type)
        
        type_layout.addStretch()
        controller_layout.addLayout(type_layout)
        
        # Spécifications temporelles
        specs_layout = QHBoxLayout()
        
        specs_layout.addWidget(QLabel("Temps de réponse (s):"))
        self.spin_settling_time = QDoubleSpinBox()
        self.spin_settling_time.setRange(0.1, 50.0)
        self.spin_settling_time.setValue(2.0)
        self.spin_settling_time.setDecimals(2)
        specs_layout.addWidget(self.spin_settling_time)
        
        specs_layout.addWidget(QLabel("Dépassement (%):"))
        self.spin_overshoot = QDoubleSpinBox()
        self.spin_overshoot.setRange(0.0, 50.0)
        self.spin_overshoot.setValue(5.0)
        self.spin_overshoot.setDecimals(1)
        specs_layout.addWidget(self.spin_overshoot)
        
        controller_layout.addLayout(specs_layout)
        
        parent_layout.addWidget(controller_group)
    
    def setup_connections(self):
        """Configuration des connexions"""
        self.matrix_editor.matrix_changed.connect(self.on_matrices_changed)
        
        # Connexions des paramètres
        self.combo_measured_outputs.currentTextChanged.connect(self.on_config_changed)
        self.spin_observer_speed.valueChanged.connect(self.on_config_changed)
        self.combo_controller_type.currentTextChanged.connect(self.on_config_changed)
        self.spin_settling_time.valueChanged.connect(self.on_config_changed)
        self.spin_overshoot.valueChanged.connect(self.on_config_changed)
    
    def on_matrices_changed(self, matrices: Dict[str, np.ndarray]):
        """Gestion du changement de matrices"""
        self.current_system.update(matrices)
        self.clear_results()
    
    def on_config_changed(self):
        """Gestion du changement de configuration"""
        self.clear_results()
    
    def clear_results(self):
        """Effacer les résultats"""
        self.results_text.clear()
        self.designed_K = None
        self.designed_L = None
    
    def verify_system_properties(self):
        """Vérifier les propriétés du système"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Toutes les matrices système requises")
            return
        
        try:
            A, B, C, D = matrices['A'], matrices['B'], matrices['C'], matrices['D']
            
            # Vérifications
            controllable = is_controllable(A, B)
            observable = is_observable(A, C)
            
            # Pôles du système
            system_poles = np.linalg.eigvals(A)
            is_stable = np.all(np.real(system_poles) < 0)
            
            # Rang des matrices
            rank_A = np.linalg.matrix_rank(A)
            rank_B = np.linalg.matrix_rank(B)
            rank_C = np.linalg.matrix_rank(C)
            
            # Affichage des résultats
            results = f"""
VÉRIFICATION DES PROPRIÉTÉS DU SYSTÈME

=== DIMENSIONS ===
• États (n): {A.shape[0]}
• Entrées (m): {B.shape[1]}
• Sorties (p): {C.shape[0]}

=== PROPRIÉTÉS STRUCTURELLES ===
• Contrôlabilité: {'✅ OUI' if controllable else '❌ NON'}
• Observabilité: {'✅ OUI' if observable else '❌ NON'}
• Stabilité: {'✅ STABLE' if is_stable else '❌ INSTABLE'}

=== RANGS DES MATRICES ===
• Rang(A): {rank_A}/{A.shape[0]}
• Rang(B): {rank_B}
• Rang(C): {rank_C}

=== PÔLES DU SYSTÈME ===
{self.format_poles_list(system_poles)}

=== FAISABILITÉ DE LA CONCEPTION ===
{self.get_design_feasibility(controllable, observable)}
            """
            
            self.results_text.setText(results.strip())
            
            if not (controllable and observable):
                QMessageBox.warning(self, "Propriétés", 
                                  "⚠️ Système non complètement contrôlable et/ou observable.\n"
                                  "La conception peut être limitée.")
            else:
                QMessageBox.information(self, "Propriétés", 
                                      "✅ Système optimal pour la conception !")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la vérification: {str(e)}")
    
    def get_design_feasibility(self, controllable: bool, observable: bool) -> str:
        """Évaluer la faisabilité de la conception"""
        if controllable and observable:
            return "✅ Conception complète possible (contrôleur + observateur)"
        elif controllable and not observable:
            return "⚠️ Observateur limité - États partiellement observables"
        elif not controllable and observable:
            return "⚠️ Contrôle limité - États partiellement contrôlables"
        else:
            return "❌ Conception très limitée - Restructuration recommandée"
    
    def format_poles_list(self, poles: np.ndarray) -> str:
        """Formatage de la liste des pôles"""
        formatted = []
        for i, pole in enumerate(poles):
            if np.isreal(pole):
                formatted.append(f"  p{i+1} = {pole.real:.4f}")
            else:
                sign = '+' if pole.imag >= 0 else '-'
                formatted.append(f"  p{i+1} = {pole.real:.4f} {sign} {abs(pole.imag):.4f}j")
        
        return '\n'.join(formatted)
    
    def design_observer(self):
        """Concevoir l'observateur de Luenberger"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'C']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et C requises")
            return
        
        try:
            A, C = matrices['A'], matrices['C']
            
            # Vérifier l'observabilité
            if not is_observable(A, C):
                QMessageBox.warning(self, "Observabilité", 
                                  "⚠️ Système non observable. L'observateur sera limité.")
            
            # Calculer les pôles désirés de l'observateur
            system_poles = np.linalg.eigvals(A)
            observer_speed = self.spin_observer_speed.value()
            
            # Pôles de l'observateur (plus rapides que le système)
            self.observer_poles = []
            for pole in system_poles:
                if np.isreal(pole):
                    obs_pole = pole.real * observer_speed
                    self.observer_poles.append(obs_pole)
                else:
                    obs_pole = pole * observer_speed
                    self.observer_poles.append(obs_pole)
            
            # Conception du gain L
            self.designed_L = design_observer(A, C, self.observer_poles)
            
            # Vérification
            A_obs = A - self.designed_L @ C
            actual_obs_poles = np.linalg.eigvals(A_obs)
            
            # Affichage des résultats
            results = f"""
CONCEPTION DE L'OBSERVATEUR DE LUENBERGER

Gain d'observateur L:
{self.format_matrix(self.designed_L)}

Pôles désirés de l'observateur:
{self.format_poles_list(np.array(self.observer_poles))}

Pôles obtenus:
{self.format_poles_list(actual_obs_poles)}

Matrice A_obs = A - L*C:
{self.format_matrix(A_obs)}

Vitesse d'observation: {observer_speed:.1f}× plus rapide que le système
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Observateur", "Observateur conçu avec succès !")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la conception de l'observateur: {str(e)}")
    
    def design_controller(self):
        """Concevoir le contrôleur"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et B requises")
            return
        
        try:
            A, B = matrices['A'], matrices['B']
            
            # Vérifier la contrôlabilité
            if not is_controllable(A, B):
                QMessageBox.warning(self, "Contrôlabilité", 
                                  "⚠️ Système non contrôlable. Le contrôleur sera limité.")
            
            # Calculer les pôles désirés du contrôleur
            tr = self.spin_settling_time.value()
            overshoot = self.spin_overshoot.value()
            
            # Calcul des paramètres du système du 2ème ordre équivalent
            if overshoot > 0:
                zeta = np.sqrt((np.log(overshoot/100))**2 / (np.pi**2 + (np.log(overshoot/100))**2))
            else:
                zeta = 1.0  # Système critique
            
            wn = 4.0 / (tr * zeta)  # Approximation pour temps de réponse à 2%
            
            # Calcul des pôles dominants
            if zeta < 1.0:
                # Pôles complexes conjugués
                real_part = -zeta * wn
                imag_part = wn * np.sqrt(1 - zeta**2)
                dominant_poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            else:
                # Pôles réels
                dominant_poles = [-zeta * wn]
            
            # Pôles supplémentaires (plus rapides)
            n = A.shape[0]
            while len(dominant_poles) < n:
                dominant_poles.append(-5.0 * wn)
            
            self.controller_poles = dominant_poles[:n]
            
            # Conception du gain K
            self.designed_K = design_state_feedback(A, B, self.controller_poles)
            
            # Vérification
            A_cl = A - B @ self.designed_K
            actual_ctrl_poles = np.linalg.eigvals(A_cl)
            
            # Affichage des résultats
            results = f"""
CONCEPTION DU CONTRÔLEUR

Spécifications:
• Temps de réponse: {tr:.2f} s
• Dépassement: {overshoot:.1f} %
• ζ = {zeta:.3f}, ωn = {wn:.3f} rad/s

Gain de contrôleur K:
{self.format_matrix(self.designed_K)}

Pôles désirés du contrôleur:
{self.format_poles_list(np.array(self.controller_poles))}

Pôles obtenus:
{self.format_poles_list(actual_ctrl_poles)}
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Contrôleur", "Contrôleur conçu avec succès !")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la conception du contrôleur: {str(e)}")
    
    def simulate_complete_system(self):
        """Simuler le système complet avec observateur et contrôleur"""
        if self.designed_K is None or self.designed_L is None:
            QMessageBox.warning(self, "Erreur", 
                              "Veuillez d'abord concevoir l'observateur ET le contrôleur")
            return
        
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Toutes les matrices système requises")
            return
        
        try:
            A, B, C, D = matrices['A'], matrices['B'], matrices['C'], matrices['D']
            
            # Paramètres de simulation
            t_final = 15.0
            dt = 0.01
            t = np.arange(0, t_final + dt, dt)
            
            # Consigne (échelon unitaire)
            r = np.ones_like(t)
            
            # Conditions initiales
            n = A.shape[0]
            x0 = np.zeros(n)  # États réels
            x_hat0 = np.random.randn(n) * 0.1  # États estimés (incertitude initiale)
            
            # Simulation du système complet
            from scipy.integrate import solve_ivp
            
            def complete_system_dynamics(t_val, state_vector):
                # state_vector = [x; x_hat]
                x = state_vector[:n]
                x_hat = state_vector[n:]
                
                # Consigne
                r_val = 1.0 if t_val >= 0 else 0.0
                
                # Sortie mesurée (avec bruit optionnel)
                y = C @ x
                if self.checkbox_add_noise.isChecked():
                    noise_level = self.spin_noise_level.value() / 100.0
                    y += np.random.randn(*y.shape) * noise_level * np.std(y)
                
                # Commande basée sur l'état estimé
                u = -self.designed_K @ x_hat + r_val
                
                # Dynamiques
                x_dot = A @ x + B @ u
                x_hat_dot = A @ x_hat + B @ u + self.designed_L @ (y - C @ x_hat)
                
                return np.concatenate([x_dot, x_hat_dot])
            
            # Conditions initiales combinées
            initial_state = np.concatenate([x0, x_hat0])
            
            # Simulation
            sol = solve_ivp(
                complete_system_dynamics,
                [0, t_final],
                initial_state,
                t_eval=t,
                dense_output=True,
                method='RK45'
            )
            
            if sol.success:
                # Extraction des résultats
                states_combined = sol.y.T
                x_real = states_combined[:, :n]
                x_estimated = states_combined[:, n:]
                
                # Calcul des sorties et commandes
                y_real = np.array([C @ x for x in x_real])
                u_sim = np.array([-self.designed_K @ x_hat + r_val 
                                for x_hat, r_val in zip(x_estimated, r)])
                
                # Erreur d'estimation
                estimation_error = x_real - x_estimated
                
                # Stocker pour visualisation
                self.simulation_data = {
                    'time': t,
                    'states_real': x_real,
                    'states_estimated': x_estimated,
                    'estimation_error': estimation_error,
                    'output': y_real,
                    'control': u_sim,
                    'reference': r
                }
                
                # Tracer automatiquement
                self.plot_complete_response()
                
                QMessageBox.information(self, "Simulation", "Simulation terminée avec succès !")
                
            else:
                QMessageBox.warning(self, "Erreur", "Échec de la simulation")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la simulation: {str(e)}")
    
    def format_matrix(self, matrix: np.ndarray) -> str:
        """Formatage d'une matrice"""
        if matrix.ndim == 1:
            return "  [" + "  ".join([f"{val:8.4f}" for val in matrix]) + "]"
        else:
            lines = []
            for row in matrix:
                lines.append("  [" + "  ".join([f"{val:8.4f}" for val in row]) + "]")
            return "\n".join(lines)
    
    def plot_complete_response(self):
        """Tracer la réponse complète du système"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Sortie et consigne
            ax.plot(data['time'], data['reference'], 'k--', linewidth=2, label='Consigne')
            
            if data['output'].ndim == 1:
                ax.plot(data['time'], data['output'], 'b-', linewidth=2, label='Sortie')
            else:
                for i in range(data['output'].shape[1]):
                    ax.plot(data['time'], data['output'][:, i], linewidth=2, label=f'Sortie {i+1}')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Réponse Complète avec Observateur')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_estimation_error(self):
        """Tracer l'erreur d'estimation"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Erreur d'estimation pour chaque état
            for i in range(data['estimation_error'].shape[1]):
                ax.plot(data['time'], data['estimation_error'][:, i], 
                       linewidth=2, label=f'Erreur état {i+1}')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Erreur d\'Estimation')
            ax.set_title('Convergence de l\'Observateur')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Ligne de référence zéro
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_states_comparison(self):
        """Comparer les états réels et estimés"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            
            # Sous-graphiques pour chaque état
            n_states = data['states_real'].shape[1]
            n_cols = min(2, n_states)
            n_rows = (n_states + n_cols - 1) // n_cols
            
            for i in range(n_states):
                ax = self.plot_canvas.figure.add_subplot(n_rows, n_cols, i+1)
                
                ax.plot(data['time'], data['states_real'][:, i], 
                       'b-', linewidth=2, label=f'État réel x{i+1}')
                ax.plot(data['time'], data['states_estimated'][:, i], 
                       'r--', linewidth=2, label=f'État estimé x̂{i+1}')
                
                ax.set_xlabel('Temps (s)')
                ax.set_ylabel(f'État {i+1}')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            self.plot_canvas.figure.suptitle('Comparaison États Réels vs Estimés')
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_controller_performance(self):
        """Tracer les performances du contrôleur"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            
            # Deux sous-graphiques
            ax1 = self.plot_canvas.figure.add_subplot(2, 1, 1)
            ax2 = self.plot_canvas.figure.add_subplot(2, 1, 2)
            
            # Graphique 1: Sortie et consigne
            ax1.plot(data['time'], data['reference'], 'k--', linewidth=2, label='Consigne')
            if data['output'].ndim == 1:
                ax1.plot(data['time'], data['output'], 'b-', linewidth=2, label='Sortie')
            else:
                for i in range(data['output'].shape[1]):
                    ax1.plot(data['time'], data['output'][:, i], linewidth=2, label=f'Sortie {i+1}')
            
            ax1.set_ylabel('Sortie')
            ax1.set_title('Performance du Contrôleur')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Graphique 2: Signal de commande
            if data['control'].ndim == 1:
                ax2.plot(data['time'], data['control'], 'r-', linewidth=2, label='Commande u')
            else:
                for i in range(data['control'].shape[1]):
                    ax2.plot(data['time'], data['control'][:, i], linewidth=2, label=f'Commande u{i+1}')
            
            ax2.set_xlabel('Temps (s)')
            ax2.set_ylabel('Commande')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def update_system(self, system_data: Dict[str, Any]):
        """Mise à jour du système depuis l'extérieur"""
        self.current_system.update(system_data)
        if 'A' in system_data:
            self.matrix_editor.set_matrices(system_data)
    
    def export_data(self):
        """Exporter les données de conception"""
        # TODO: Implémenter l'export
        QMessageBox.information(self, "Export", "Export des données de conception à implémenter")
    
    def capture_figure(self):
        """Capturer la figure actuelle"""
        self.plot_canvas.save_figure()
