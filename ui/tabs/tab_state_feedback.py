#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet Commande par Rétroaction d'État
Conception de régulateurs par placement de pôles et feedback d'état
"""

import numpy as np
from typing import Dict, Any, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTextEdit, QMessageBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..widgets_common import MatrixEditor, PlotCanvas, ParameterPanel
from ...core.place_design import design_state_feedback, place_poles
from ...core.lti_tools import time_response
from ...core.ctrb_obsv import is_controllable
from ...core.utils import validate_system_matrices


class TabStateFeedback(QWidget):
    """Onglet de conception de commande par rétroaction d'état"""
    
    system_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.designed_K = None
        self.desired_poles = []
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
        
        # Spécifications de conception
        self.setup_design_specifications(left_layout)
        
        # Boutons de conception
        design_group = QGroupBox("Conception du Régulateur")
        design_layout = QVBoxLayout(design_group)
        
        self.btn_check_controllability = QPushButton("Vérifier Contrôlabilité")
        self.btn_check_controllability.setObjectName("actionButton")
        self.btn_check_controllability.clicked.connect(self.check_controllability)
        design_layout.addWidget(self.btn_check_controllability)
        
        self.btn_design_K = QPushButton("Concevoir Gain K")
        self.btn_design_K.setObjectName("validateButton")
        self.btn_design_K.clicked.connect(self.design_feedback_gain)
        design_layout.addWidget(self.btn_design_K)
        
        self.btn_simulate = QPushButton("Simuler Boucle Fermée")
        self.btn_simulate.setObjectName("actionButton")
        self.btn_simulate.clicked.connect(self.simulate_closed_loop)
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
        self.plot_canvas = PlotCanvas("Réponse en Boucle Fermée")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de visualisation
        viz_buttons_layout = QHBoxLayout()
        
        viz_buttons = [
            ("Réponse Temporelle", self.plot_time_response),
            ("Évolution États", self.plot_state_evolution),
            ("Signal de Commande", self.plot_control_signal),
            ("Lieu des Pôles", self.plot_pole_placement)
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
        
    def setup_design_specifications(self, parent_layout):
        """Configuration des spécifications de conception"""
        specs_group = QGroupBox("Spécifications de Performance")
        specs_layout = QVBoxLayout(specs_group)
        
        # Méthode de spécification
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Méthode:"))
        
        self.btn_specs_method = QPushButton("Spécifications Temporelles")
        self.btn_specs_method.setCheckable(True)
        self.btn_specs_method.setChecked(True)
        self.btn_specs_method.clicked.connect(self.toggle_specification_method)
        method_layout.addWidget(self.btn_specs_method)
        
        self.btn_poles_method = QPushButton("Pôles Directs")
        self.btn_poles_method.setCheckable(True)
        self.btn_poles_method.clicked.connect(self.toggle_specification_method)
        method_layout.addWidget(self.btn_poles_method)
        
        method_layout.addStretch()
        specs_layout.addLayout(method_layout)
        
        # Zone des spécifications temporelles
        self.specs_temporal = self.create_temporal_specs()
        specs_layout.addWidget(self.specs_temporal)
        
        # Zone des pôles désirés
        self.specs_poles = self.create_poles_specs()
        self.specs_poles.setVisible(False)
        specs_layout.addWidget(self.specs_poles)
        
        parent_layout.addWidget(specs_group)
    
    def create_temporal_specs(self) -> QGroupBox:
        """Créer la zone des spécifications temporelles"""
        group = QGroupBox("Spécifications Temporelles")
        layout = QVBoxLayout(group)
        
        # Paramètres temporels
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Temps de réponse (s):"))
        self.spin_tr = QDoubleSpinBox()
        self.spin_tr.setRange(0.1, 100.0)
        self.spin_tr.setValue(2.0)
        self.spin_tr.setDecimals(2)
        params_layout.addWidget(self.spin_tr)
        
        params_layout.addWidget(QLabel("Dépassement (%):"))
        self.spin_overshoot = QDoubleSpinBox()
        self.spin_overshoot.setRange(0.0, 100.0)
        self.spin_overshoot.setValue(5.0)
        self.spin_overshoot.setDecimals(1)
        params_layout.addWidget(self.spin_overshoot)
        
        layout.addLayout(params_layout)
        
        # Calcul automatique des pôles
        self.btn_compute_poles = QPushButton("Calculer Pôles Désirés")
        self.btn_compute_poles.setObjectName("actionButton")
        self.btn_compute_poles.clicked.connect(self.compute_desired_poles)
        layout.addWidget(self.btn_compute_poles)
        
        return group
    
    def create_poles_specs(self) -> QGroupBox:
        """Créer la zone des pôles désirés"""
        group = QGroupBox("Pôles Désirés")
        layout = QVBoxLayout(group)
        
        # Nombre de pôles
        poles_count_layout = QHBoxLayout()
        poles_count_layout.addWidget(QLabel("Nombre de pôles:"))
        
        self.spin_poles_count = QSpinBox()
        self.spin_poles_count.setRange(1, 10)
        self.spin_poles_count.setValue(2)
        self.spin_poles_count.valueChanged.connect(self.update_poles_table)
        poles_count_layout.addWidget(self.spin_poles_count)
        
        poles_count_layout.addStretch()
        layout.addLayout(poles_count_layout)
        
        # Table des pôles
        self.poles_table = QTableWidget(2, 2)
        self.poles_table.setHorizontalHeaderLabels(["Partie Réelle", "Partie Imaginaire"])
        self.poles_table.verticalHeader().setVisible(False)
        layout.addWidget(self.poles_table)
        
        # Initialiser la table
        self.update_poles_table()
        
        return group
    
    def setup_connections(self):
        """Configuration des connexions"""
        self.matrix_editor.matrix_changed.connect(self.on_matrices_changed)
        
        # Connexions des spécifications
        self.spin_tr.valueChanged.connect(self.on_specs_changed)
        self.spin_overshoot.valueChanged.connect(self.on_specs_changed)
    
    def toggle_specification_method(self):
        """Basculer entre méthodes de spécification"""
        sender = self.sender()
        
        if sender == self.btn_specs_method:
            self.btn_specs_method.setChecked(True)
            self.btn_poles_method.setChecked(False)
            self.specs_temporal.setVisible(True)
            self.specs_poles.setVisible(False)
        else:
            self.btn_specs_method.setChecked(False)
            self.btn_poles_method.setChecked(True)
            self.specs_temporal.setVisible(False)
            self.specs_poles.setVisible(True)
    
    def update_poles_table(self):
        """Mise à jour de la table des pôles"""
        n_poles = self.spin_poles_count.value()
        self.poles_table.setRowCount(n_poles)
        
        # Initialiser avec des valeurs par défaut
        for i in range(n_poles):
            if self.poles_table.item(i, 0) is None:
                self.poles_table.setItem(i, 0, QTableWidgetItem(f"{-1.0 - i * 0.5:.2f}"))
            if self.poles_table.item(i, 1) is None:
                self.poles_table.setItem(i, 1, QTableWidgetItem("0.0"))
    
    def on_matrices_changed(self, matrices: Dict[str, np.ndarray]):
        """Gestion du changement de matrices"""
        self.current_system.update(matrices)
        
        # Mettre à jour le nombre de pôles basé sur la dimension de A
        if 'A' in matrices and matrices['A'] is not None:
            n = matrices['A'].shape[0]
            self.spin_poles_count.setValue(n)
        
        self.clear_results()
    
    def on_specs_changed(self):
        """Gestion du changement de spécifications"""
        # Recalculer automatiquement si en mode spécifications temporelles
        if self.btn_specs_method.isChecked():
            self.compute_desired_poles()
    
    def clear_results(self):
        """Effacer les résultats"""
        self.results_text.clear()
        self.designed_K = None
    
    def compute_desired_poles(self):
        """Calculer les pôles désirés à partir des spécifications temporelles"""
        tr = self.spin_tr.value()
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
        
        # Mise à jour de l'affichage
        matrices = {k: v for k, v in self.current_system.items() if k in ['A']}
        if 'A' in matrices and matrices['A'] is not None:
            n = matrices['A'].shape[0]
            
            # Pôles supplémentaires (plus rapides)
            while len(dominant_poles) < n:
                dominant_poles.append(-5.0 * wn)  # Pôles 5 fois plus rapides
            
            self.desired_poles = dominant_poles[:n]
            
            # Affichage dans les résultats
            results = f"""
PÔLES DÉSIRÉS CALCULÉS

Spécifications:
- Temps de réponse: {tr:.2f} s
- Dépassement: {overshoot:.1f} %

Paramètres calculés:
- Amortissement ζ: {zeta:.3f}
- Pulsation naturelle ωn: {wn:.3f} rad/s

Pôles désirés:
{self.format_poles(self.desired_poles)}
            """
            
            self.results_text.setText(results.strip())
    
    def get_desired_poles_from_table(self) -> List[complex]:
        """Récupérer les pôles désirés depuis la table"""
        poles = []
        for i in range(self.poles_table.rowCount()):
            try:
                real_item = self.poles_table.item(i, 0)
                imag_item = self.poles_table.item(i, 1)
                
                if real_item and imag_item:
                    real_part = float(real_item.text())
                    imag_part = float(imag_item.text())
                    poles.append(complex(real_part, imag_part))
            except ValueError:
                pass
        
        return poles
    
    def format_poles(self, poles: List[complex]) -> str:
        """Formatage de la liste des pôles"""
        formatted = []
        for i, pole in enumerate(poles):
            if np.isreal(pole):
                formatted.append(f"  p{i+1} = {pole.real:.4f}")
            else:
                sign = '+' if pole.imag >= 0 else '-'
                formatted.append(f"  p{i+1} = {pole.real:.4f} {sign} {abs(pole.imag):.4f}j")
        
        return '\n'.join(formatted)
    
    def check_controllability(self):
        """Vérifier la contrôlabilité du système"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et B requises")
            return
        
        try:
            A, B = matrices['A'], matrices['B']
            controllable = is_controllable(A, B)
            
            if controllable:
                QMessageBox.information(self, "Contrôlabilité", 
                                      "✓ Le système est contrôlable.\nLe placement de pôles est possible.")
            else:
                QMessageBox.warning(self, "Contrôlabilité", 
                                  "✗ Le système n'est pas contrôlable.\nLe placement de pôles n'est pas possible.")
            
            return controllable
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la vérification: {str(e)}")
            return False
    
    def design_feedback_gain(self):
        """Concevoir le gain de rétroaction K"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices A et B requises")
            return
        
        # Vérifier la contrôlabilité d'abord
        if not self.check_controllability():
            return
        
        try:
            A, B = matrices['A'], matrices['B']
            
            # Récupérer les pôles désirés
            if self.btn_specs_method.isChecked():
                if not hasattr(self, 'desired_poles') or not self.desired_poles:
                    self.compute_desired_poles()
                desired_poles = self.desired_poles
            else:
                desired_poles = self.get_desired_poles_from_table()
            
            if len(desired_poles) != A.shape[0]:
                QMessageBox.warning(self, "Erreur", 
                                  f"Nombre de pôles ({len(desired_poles)}) différent de la dimension du système ({A.shape[0]})")
                return
            
            # Conception du gain K
            self.designed_K = design_state_feedback(A, B, desired_poles)
            
            # Vérification : calcul des pôles en boucle fermée
            A_cl = A - B @ self.designed_K
            actual_poles = np.linalg.eigvals(A_cl)
            
            # Affichage des résultats
            results = f"""
CONCEPTION DU RÉGULATEUR PAR RÉTROACTION D'ÉTAT

Gain de rétroaction K:
{self.format_matrix(self.designed_K)}

Pôles désirés:
{self.format_poles(desired_poles)}

Pôles obtenus en boucle fermée:
{self.format_poles(actual_poles.tolist())}

Erreur de placement:
{self.compute_pole_placement_error(desired_poles, actual_poles)}

Matrice A en boucle fermée:
{self.format_matrix(A_cl)}
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Conception", "Gain K conçu avec succès !")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la conception: {str(e)}")
    
    def format_matrix(self, matrix: np.ndarray) -> str:
        """Formatage d'une matrice"""
        if matrix.ndim == 1:
            return "  [" + "  ".join([f"{val:8.4f}" for val in matrix]) + "]"
        else:
            lines = []
            for row in matrix:
                lines.append("  [" + "  ".join([f"{val:8.4f}" for val in row]) + "]")
            return "\n".join(lines)
    
    def compute_pole_placement_error(self, desired: List[complex], actual: np.ndarray) -> str:
        """Calculer l'erreur de placement de pôles"""
        # Trier les pôles pour comparaison
        desired_sorted = sorted(desired, key=lambda x: (x.real, x.imag))
        actual_sorted = sorted(actual, key=lambda x: (x.real, x.imag))
        
        errors = []
        for i, (d, a) in enumerate(zip(desired_sorted, actual_sorted)):
            error = abs(d - a)
            errors.append(f"  Pôle {i+1}: {error:.6f}")
        
        max_error = max([abs(d - a) for d, a in zip(desired_sorted, actual_sorted)])
        errors.append(f"  Erreur maximale: {max_error:.6f}")
        
        return "\n".join(errors)
    
    def simulate_closed_loop(self):
        """Simuler la réponse en boucle fermée"""
        if self.designed_K is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord concevoir le gain K")
            return
        
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Toutes les matrices système requises")
            return
        
        try:
            A, B, C, D = matrices['A'], matrices['B'], matrices['C'], matrices['D']
            
            # Système en boucle fermée
            A_cl = A - B @ self.designed_K
            B_cl = B  # Pour la consigne
            
            # Paramètres de simulation
            t_final = 10.0
            dt = 0.01
            t = np.arange(0, t_final + dt, dt)
            
            # Consigne (échelon unitaire)
            r = np.ones_like(t)
            
            # Simulation
            from scipy.integrate import solve_ivp
            
            def closed_loop_dynamics(t, x, K, A, B, r_func):
                r_val = r_func(t)
                u = -K @ x + r_val  # Commande avec consigne
                return A @ x + B @ u
            
            # Fonction de consigne
            r_func = lambda t_val: 1.0 if t_val >= 0 else 0.0
            
            # Simulation
            sol = solve_ivp(
                lambda t, x: closed_loop_dynamics(t, x, self.designed_K, A, B, r_func),
                [0, t_final],
                np.zeros(A.shape[0]),  # Conditions initiales nulles
                t_eval=t,
                dense_output=True
            )
            
            if sol.success:
                x_sim = sol.y.T
                y_sim = x_sim @ C.T + (r[:, np.newaxis] @ D)
                u_sim = np.array([-self.designed_K @ x + r_val for x, r_val in zip(x_sim, r)])
                
                # Stocker pour visualisation
                self.simulation_data = {
                    'time': t,
                    'states': x_sim,
                    'output': y_sim,
                    'control': u_sim,
                    'reference': r
                }
                
                # Tracer automatiquement
                self.plot_time_response()
                
            else:
                QMessageBox.warning(self, "Erreur", "Échec de la simulation")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la simulation: {str(e)}")
    
    def plot_time_response(self):
        """Tracer la réponse temporelle"""
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
            ax.set_title('Réponse en Boucle Fermée')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_state_evolution(self):
        """Tracer l'évolution des états"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Évolution de tous les états
            for i in range(data['states'].shape[1]):
                ax.plot(data['time'], data['states'][:, i], linewidth=2, label=f'État x{i+1}')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Valeur des États')
            ax.set_title('Évolution des États en Boucle Fermée')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_control_signal(self):
        """Tracer le signal de commande"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Signal de commande
            if data['control'].ndim == 1:
                ax.plot(data['time'], data['control'], 'r-', linewidth=2, label='Commande u')
            else:
                for i in range(data['control'].shape[1]):
                    ax.plot(data['time'], data['control'][:, i], linewidth=2, label=f'Commande u{i+1}')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Signal de Commande')
            ax.set_title('Signal de Commande en Boucle Fermée')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_pole_placement(self):
        """Tracer le lieu des pôles"""
        if self.designed_K is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord concevoir le gain K")
            return
        
        try:
            matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B']}
            A, B = matrices['A'], matrices['B']
            
            # Pôles en boucle ouverte
            poles_open = np.linalg.eigvals(A)
            
            # Pôles en boucle fermée
            A_cl = A - B @ self.designed_K
            poles_closed = np.linalg.eigvals(A_cl)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Pôles boucle ouverte
            ax.scatter(np.real(poles_open), np.imag(poles_open), 
                      marker='x', s=100, c='red', linewidth=3, label='Boucle ouverte')
            
            # Pôles boucle fermée
            ax.scatter(np.real(poles_closed), np.imag(poles_closed), 
                      marker='o', s=100, c='blue', linewidth=2, label='Boucle fermée')
            
            # Ligne de stabilité
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Limite de stabilité')
            
            ax.set_xlabel('Partie Réelle')
            ax.set_ylabel('Partie Imaginaire')
            ax.set_title('Placement des Pôles')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
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
