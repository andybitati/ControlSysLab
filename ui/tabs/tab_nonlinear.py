#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet Systèmes Non-Linéaires
Analyse de systèmes non-linéaires, linéarisation et méthode de Lyapunov
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, List, Callable, Tuple, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTextEdit, QMessageBox, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..widgets_common import PlotCanvas, ParameterPanel
from ...core.nonlinear import (
    linearize_system, lyapunov_stability_analysis, simulate_nonlinear_system,
    phase_portrait, lyapunov_function_quadratic, evaluate_lyapunov_derivative
)
from ...core.utils import validate_nonlinear_system


class TabNonlinear(QWidget):
    """Onglet d'analyse des systèmes non-linéaires"""
    
    system_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.nonlinear_function = None
        self.equilibrium_point = None
        self.linearized_system = None
        self.lyapunov_data = None
        self.setup_ui()
        self.setup_connections()
        self.setup_predefined_systems()
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Zone de gauche : définition du système et paramètres
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Définition du système non-linéaire
        self.setup_system_definition(left_layout)
        
        # Analyse de linéarisation
        self.setup_linearization_analysis(left_layout)
        
        # Analyse de Lyapunov
        self.setup_lyapunov_analysis(left_layout)
        
        # Paramètres de simulation
        self.param_panel = ParameterPanel("Paramètres de Simulation")
        self.param_panel.add_parameter("t_final", "Temps final", 20.0, 1.0, 100.0, 1, "s")
        self.param_panel.add_parameter("dt", "Pas de temps", 0.01, 0.001, 1.0, 3, "s")
        self.param_panel.add_parameter("x1_range", "Plage x1", 5.0, 1.0, 20.0, 1)
        self.param_panel.add_parameter("x2_range", "Plage x2", 5.0, 1.0, 20.0, 1)
        left_layout.addWidget(self.param_panel)
        
        # Zone de résultats textuels
        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        self.results_text.setMaximumHeight(200)
        left_layout.addWidget(self.results_text)
        
        main_splitter.addWidget(left_widget)
        
        # Zone de droite : visualisations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Canvas pour les graphiques
        self.plot_canvas = PlotCanvas("Analyse Non-Linéaire")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de visualisation
        viz_buttons_layout = QHBoxLayout()
        
        viz_buttons = [
            ("Portrait de Phase", self.plot_phase_portrait),
            ("Trajectoires", self.plot_trajectories),
            ("Fonction Lyapunov", self.plot_lyapunov_function),
            ("Comparaison Lin./NonLin.", self.plot_comparison)
        ]
        
        for text, callback in viz_buttons:
            btn = QPushButton(text)
            btn.setObjectName("plotButton")
            btn.clicked.connect(callback)
            viz_buttons_layout.addWidget(btn)
        
        right_layout.addLayout(viz_buttons_layout)
        main_splitter.addWidget(right_widget)
        
        # Proportions du splitter
        main_splitter.setSizes([600, 600])
    
    def setup_system_definition(self, parent_layout):
        """Configuration de la définition du système"""
        system_group = QGroupBox("Définition du Système Non-Linéaire")
        system_layout = QVBoxLayout(system_group)
        
        # Sélection de système prédéfini
        predefined_layout = QHBoxLayout()
        predefined_layout.addWidget(QLabel("Système prédéfini:"))
        
        self.combo_predefined = QComboBox()
        self.combo_predefined.addItems([
            "Personnalisé",
            "Pendule simple",
            "Van der Pol",
            "Lorenz",
            "Duffing",
            "Système proie-prédateur"
        ])
        self.combo_predefined.currentTextChanged.connect(self.load_predefined_system)
        predefined_layout.addWidget(self.combo_predefined)
        
        predefined_layout.addStretch()
        system_layout.addLayout(predefined_layout)
        
        # Éditeur d'équations
        equations_layout = QVBoxLayout()
        equations_layout.addWidget(QLabel("Équations différentielles (format Python/SymPy):"))
        
        self.text_equations = QTextEdit()
        self.text_equations.setMaximumHeight(100)
        self.text_equations.setPlainText("# dx1/dt = x2\n# dx2/dt = -sin(x1) - 0.1*x2")
        self.text_equations.setObjectName("codeEditor")
        equations_layout.addWidget(self.text_equations)
        
        system_layout.addLayout(equations_layout)
        
        # Point d'équilibre
        equilibrium_layout = QHBoxLayout()
        equilibrium_layout.addWidget(QLabel("Point d'équilibre [x1, x2]:"))
        
        self.line_equilibrium = QLineEdit("[0, 0]")
        self.line_equilibrium.setMaximumWidth(150)
        equilibrium_layout.addWidget(self.line_equilibrium)
        
        equilibrium_layout.addStretch()
        system_layout.addLayout(equilibrium_layout)
        
        # Boutons d'action
        actions_layout = QHBoxLayout()
        
        self.btn_validate_system = QPushButton("Valider Système")
        self.btn_validate_system.setObjectName("actionButton")
        self.btn_validate_system.clicked.connect(self.validate_nonlinear_system)
        actions_layout.addWidget(self.btn_validate_system)
        
        self.btn_linearize = QPushButton("Linéariser")
        self.btn_linearize.setObjectName("actionButton")
        self.btn_linearize.clicked.connect(self.linearize_system)
        actions_layout.addWidget(self.btn_linearize)
        
        self.btn_simulate = QPushButton("Simuler")
        self.btn_simulate.setObjectName("validateButton")
        self.btn_simulate.clicked.connect(self.simulate_system)
        actions_layout.addWidget(self.btn_simulate)
        
        actions_layout.addStretch()
        system_layout.addLayout(actions_layout)
        
        parent_layout.addWidget(system_group)
    
    def setup_linearization_analysis(self, parent_layout):
        """Configuration de l'analyse de linéarisation"""
        linear_group = QGroupBox("Analyse de Linéarisation")
        linear_layout = QVBoxLayout(linear_group)
        
        # Options d'analyse
        options_layout = QHBoxLayout()
        
        self.checkbox_auto_linearize = QCheckBox("Linéarisation automatique")
        self.checkbox_auto_linearize.setChecked(True)
        options_layout.addWidget(self.checkbox_auto_linearize)
        
        self.checkbox_show_eigenvals = QCheckBox("Afficher valeurs propres")
        self.checkbox_show_eigenvals.setChecked(True)
        options_layout.addWidget(self.checkbox_show_eigenvals)
        
        options_layout.addStretch()
        linear_layout.addLayout(options_layout)
        
        parent_layout.addWidget(linear_group)
    
    def setup_lyapunov_analysis(self, parent_layout):
        """Configuration de l'analyse de Lyapunov"""
        lyapunov_group = QGroupBox("Analyse de Stabilité de Lyapunov")
        lyapunov_layout = QVBoxLayout(lyapunov_group)
        
        # Type d'analyse
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Méthode:"))
        
        self.combo_lyapunov_method = QComboBox()
        self.combo_lyapunov_method.addItems([
            "Automatique (Lyapunov linéarisé)",
            "Fonction quadratique V = x'Px",
            "Fonction personnalisée"
        ])
        type_layout.addWidget(self.combo_lyapunov_method)
        
        type_layout.addStretch()
        lyapunov_layout.addLayout(type_layout)
        
        # Fonction de Lyapunov personnalisée
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("Fonction V(x) personnalisée (optionnel):"))
        
        self.text_lyapunov_function = QLineEdit("x1**2 + x2**2")
        self.text_lyapunov_function.setEnabled(False)
        custom_layout.addWidget(self.text_lyapunov_function)
        
        lyapunov_layout.addLayout(custom_layout)
        
        # Bouton d'analyse
        self.btn_lyapunov_analysis = QPushButton("Analyser Stabilité Lyapunov")
        self.btn_lyapunov_analysis.setObjectName("actionButton")
        self.btn_lyapunov_analysis.clicked.connect(self.analyze_lyapunov_stability)
        lyapunov_layout.addWidget(self.btn_lyapunov_analysis)
        
        # Connexion pour activer/désactiver fonction personnalisée
        self.combo_lyapunov_method.currentTextChanged.connect(
            lambda text: self.text_lyapunov_function.setEnabled("personnalisée" in text)
        )
        
        parent_layout.addWidget(lyapunov_group)
    
    def setup_predefined_systems(self):
        """Configuration des systèmes prédéfinis"""
        self.predefined_systems = {
            "Pendule simple": {
                "equations": "# Pendule simple avec amortissement\n# dx1/dt = x2\n# dx2/dt = -sin(x1) - 0.1*x2",
                "equilibrium": "[0, 0]",
                "description": "Pendule simple avec amortissement visqueux"
            },
            "Van der Pol": {
                "equations": "# Oscillateur de Van der Pol\n# dx1/dt = x2\n# dx2/dt = mu*(1-x1**2)*x2 - x1",
                "equilibrium": "[0, 0]",
                "description": "Oscillateur auto-entretenu de Van der Pol"
            },
            "Lorenz": {
                "equations": "# Système de Lorenz (version 2D simplifiée)\n# dx1/dt = sigma*(x2-x1)\n# dx2/dt = x1*(rho-x1) - x2",
                "equilibrium": "[0, 0]",
                "description": "Version 2D du système de Lorenz"
            },
            "Duffing": {
                "equations": "# Oscillateur de Duffing\n# dx1/dt = x2\n# dx2/dt = -delta*x2 - alpha*x1 - beta*x1**3",
                "equilibrium": "[0, 0]",
                "description": "Oscillateur de Duffing avec non-linéarité cubique"
            },
            "Système proie-prédateur": {
                "equations": "# Modèle proie-prédateur de Lotka-Volterra modifié\n# dx1/dt = x1*(alpha - beta*x2)\n# dx2/dt = x2*(-gamma + delta*x1)",
                "equilibrium": "[1, 1]",
                "description": "Modèle proie-prédateur avec point d'équilibre non-trivial"
            }
        }
    
    def setup_connections(self):
        """Configuration des connexions"""
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)
    
    def on_parameters_changed(self, parameters: Dict[str, float]):
        """Gestion du changement de paramètres"""
        self.current_system.update(parameters)
    
    def load_predefined_system(self, system_name: str):
        """Charger un système prédéfini"""
        if system_name == "Personnalisé":
            return
        
        if system_name in self.predefined_systems:
            system_data = self.predefined_systems[system_name]
            self.text_equations.setPlainText(system_data["equations"])
            self.line_equilibrium.setText(system_data["equilibrium"])
            
            QMessageBox.information(self, "Système chargé", 
                                  f"Système '{system_name}' chargé:\n{system_data['description']}")
    
    def validate_nonlinear_system(self):
        """Valider le système non-linéaire"""
        try:
            equations_text = self.text_equations.toPlainText()
            equilibrium_text = self.line_equilibrium.text()
            
            # Parse du point d'équilibre
            self.equilibrium_point = eval(equilibrium_text)
            if not isinstance(self.equilibrium_point, (list, tuple)):
                raise ValueError("Le point d'équilibre doit être une liste ou tuple")
            
            self.equilibrium_point = np.array(self.equilibrium_point)
            
            # Validation et compilation des équations
            is_valid, compiled_func, error_msg = validate_nonlinear_system(equations_text)
            
            if is_valid:
                self.nonlinear_function = compiled_func
                self.results_text.setText(f"""
SYSTÈME NON-LINÉAIRE VALIDÉ

Équations: 
{equations_text}

Point d'équilibre: {self.equilibrium_point}

✅ Système prêt pour l'analyse !
                """.strip())
                
                QMessageBox.information(self, "Validation", "Système non-linéaire validé avec succès !")
            else:
                self.results_text.setText(f"❌ Erreur de validation: {error_msg}")
                QMessageBox.warning(self, "Erreur de validation", error_msg)
                
        except Exception as e:
            error_msg = f"Erreur lors de la validation: {str(e)}"
            self.results_text.setText(f"❌ {error_msg}")
            QMessageBox.warning(self, "Erreur", error_msg)
    
    def linearize_system(self):
        """Linéariser le système autour du point d'équilibre"""
        if self.nonlinear_function is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord valider le système non-linéaire")
            return
        
        try:
            # Linéarisation
            A, B = linearize_system(self.nonlinear_function, self.equilibrium_point)
            
            # Analyse du système linéarisé
            eigenvals = np.linalg.eigvals(A)
            is_stable = np.all(np.real(eigenvals) < 0)
            
            # Sauvegarde
            self.linearized_system = {'A': A, 'B': B, 'eigenvals': eigenvals}
            
            # Affichage des résultats
            results = f"""
LINÉARISATION DU SYSTÈME

Point d'équilibre: {self.equilibrium_point}

Matrice A (jacobienne par rapport à x):
{self.format_matrix(A)}

Matrice B (jacobienne par rapport à u):
{self.format_matrix(B)}

Valeurs propres du système linéarisé:
{self.format_eigenvalues(eigenvals)}

Stabilité locale: {'✅ STABLE' if is_stable else '❌ INSTABLE'}

Type de point d'équilibre: {self.classify_equilibrium_point(eigenvals)}
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Linéarisation", "Linéarisation effectuée avec succès !")
            
        except Exception as e:
            error_msg = f"Erreur lors de la linéarisation: {str(e)}"
            self.results_text.setText(f"❌ {error_msg}")
            QMessageBox.warning(self, "Erreur", error_msg)
    
    def analyze_lyapunov_stability(self):
        """Analyser la stabilité par la méthode de Lyapunov"""
        if self.linearized_system is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord linéariser le système")
            return
        
        try:
            method = self.combo_lyapunov_method.currentText()
            A = self.linearized_system['A']
            
            if "Automatique" in method:
                # Méthode automatique basée sur le système linéarisé
                P, is_stable, lyap_func = lyapunov_function_quadratic(A)
                
                if P is not None:
                    self.lyapunov_data = {
                        'P': P,
                        'function': lyap_func,
                        'stable': is_stable,
                        'method': 'quadratic_auto'
                    }
                    
                    results = f"""
ANALYSE DE STABILITÉ DE LYAPUNOV

Méthode: Fonction quadratique automatique V(x) = x'Px

Matrice P:
{self.format_matrix(P)}

Stabilité: {'✅ STABLE' if is_stable else '❌ INSTABLE'}

Fonction de Lyapunov: V(x) = x₁²·P₁₁ + 2x₁x₂·P₁₂ + x₂²·P₂₂

Condition de stabilité: V̇(x) < 0 pour x ≠ 0
                    """
                else:
                    results = "❌ Impossible de construire une fonction de Lyapunov quadratique"
                
            elif "quadratique" in method:
                # Méthode quadratique manuelle
                P = np.eye(len(self.equilibrium_point))  # Matrice identité par défaut
                lyap_func = lambda x: x.T @ P @ x
                
                # Vérification de la stabilité
                derivative_func = lambda x: evaluate_lyapunov_derivative(
                    lyap_func, self.nonlinear_function, x
                )
                
                self.lyapunov_data = {
                    'P': P,
                    'function': lyap_func,
                    'derivative': derivative_func,
                    'method': 'quadratic_manual'
                }
                
                results = f"""
ANALYSE DE STABILITÉ DE LYAPUNOV

Méthode: Fonction quadratique V(x) = x'Px

Matrice P (identité):
{self.format_matrix(P)}

Note: Vérification graphique de V̇(x) recommandée
                """
            
            else:
                # Fonction personnalisée
                lyap_expr = self.text_lyapunov_function.text()
                # TODO: Implémenter l'analyse avec fonction personnalisée
                results = "⚠️ Fonction personnalisée - À implémenter"
            
            self.results_text.setText(results.strip())
            
        except Exception as e:
            error_msg = f"Erreur lors de l'analyse de Lyapunov: {str(e)}"
            self.results_text.setText(f"❌ {error_msg}")
            QMessageBox.warning(self, "Erreur", error_msg)
    
    def simulate_system(self):
        """Simuler le système non-linéaire"""
        if self.nonlinear_function is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord valider le système non-linéaire")
            return
        
        try:
            # Paramètres de simulation
            params = self.param_panel.get_parameters()
            t_final = params.get('t_final', 20.0)
            dt = params.get('dt', 0.01)
            
            # Conditions initiales multiples pour trajectoires variées
            initial_conditions = [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.5, 1.0],
                [-0.5, -1.0],
                [2.0, 0.5]
            ]
            
            # Simulation
            simulation_results = []
            t = np.arange(0, t_final + dt, dt)
            
            for x0 in initial_conditions:
                sol = simulate_nonlinear_system(
                    self.nonlinear_function, x0, t_final, dt
                )
                if sol.success:
                    simulation_results.append({
                        'initial': x0,
                        'time': sol.t,
                        'trajectory': sol.y.T
                    })
            
            # Sauvegarde pour visualisation
            self.simulation_data = {
                'results': simulation_results,
                'time_vector': t,
                'parameters': params
            }
            
            # Tracer automatiquement
            self.plot_phase_portrait()
            
            QMessageBox.information(self, "Simulation", 
                                  f"Simulation terminée avec succès !\n"
                                  f"{len(simulation_results)} trajectoires calculées.")
            
        except Exception as e:
            error_msg = f"Erreur lors de la simulation: {str(e)}"
            QMessageBox.warning(self, "Erreur", error_msg)
    
    def format_matrix(self, matrix: np.ndarray) -> str:
        """Formatage d'une matrice"""
        if matrix.ndim == 1:
            return "  [" + "  ".join([f"{val:8.4f}" for val in matrix]) + "]"
        else:
            lines = []
            for row in matrix:
                lines.append("  [" + "  ".join([f"{val:8.4f}" for val in row]) + "]")
            return "\n".join(lines)
    
    def format_eigenvalues(self, eigenvals: np.ndarray) -> str:
        """Formatage des valeurs propres"""
        formatted = []
        for i, val in enumerate(eigenvals):
            if np.isreal(val):
                formatted.append(f"  λ{i+1} = {val.real:.4f}")
            else:
                sign = '+' if val.imag >= 0 else '-'
                formatted.append(f"  λ{i+1} = {val.real:.4f} {sign} {abs(val.imag):.4f}j")
        
        return '\n'.join(formatted)
    
    def classify_equilibrium_point(self, eigenvals: np.ndarray) -> str:
        """Classification du point d'équilibre"""
        real_parts = np.real(eigenvals)
        imag_parts = np.imag(eigenvals)
        
        if np.all(real_parts < 0):
            if np.any(imag_parts != 0):
                return "Foyer stable (spirale convergente)"
            else:
                return "Nœud stable"
        elif np.all(real_parts > 0):
            if np.any(imag_parts != 0):
                return "Foyer instable (spirale divergente)"
            else:
                return "Nœud instable"
        elif np.any(real_parts > 0) and np.any(real_parts < 0):
            return "Point de selle (instable)"
        else:
            return "Centre ou cas dégénéré"
    
    def plot_phase_portrait(self):
        """Tracer le portrait de phase"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            params = self.param_panel.get_parameters()
            x1_range = params.get('x1_range', 5.0)
            x2_range = params.get('x2_range', 5.0)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Champ de vecteurs
            x1_grid = np.linspace(-x1_range, x1_range, 20)
            x2_grid = np.linspace(-x2_range, x2_range, 20)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            
            DX1 = np.zeros_like(X1)
            DX2 = np.zeros_like(X2)
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    state = np.array([X1[i, j], X2[i, j]])
                    derivatives = self.nonlinear_function(0, state)
                    DX1[i, j] = derivatives[0]
                    DX2[i, j] = derivatives[1]
            
            # Normalisation pour meilleur affichage
            M = np.sqrt(DX1**2 + DX2**2)
            M[M == 0] = 1
            DX1_norm = DX1 / M
            DX2_norm = DX2 / M
            
            # Tracé du champ de vecteurs
            ax.quiver(X1, X2, DX1_norm, DX2_norm, M, 
                     cmap='Blues', alpha=0.6, scale=30)
            
            # Trajectoires
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, result in enumerate(self.simulation_data['results']):
                color = colors[i % len(colors)]
                traj = result['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, 
                       label=f"CI: {result['initial']}")
                
                # Point initial
                ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)
                
                # Point final
                ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=6)
            
            # Point d'équilibre
            if self.equilibrium_point is not None:
                ax.plot(self.equilibrium_point[0], self.equilibrium_point[1], 
                       'ko', markersize=10, markerfacecolor='yellow', 
                       markeredgewidth=2, label='Équilibre')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('Portrait de Phase')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(-x1_range, x1_range)
            ax.set_ylim(-x2_range, x2_range)
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du tracé: {str(e)}")
    
    def plot_trajectories(self):
        """Tracer les trajectoires temporelles"""
        if not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            self.plot_canvas.clear_plot()
            
            # Deux sous-graphiques
            ax1 = self.plot_canvas.figure.add_subplot(2, 1, 1)
            ax2 = self.plot_canvas.figure.add_subplot(2, 1, 2)
            
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            
            for i, result in enumerate(self.simulation_data['results']):
                color = colors[i % len(colors)]
                time = result['time']
                traj = result['trajectory']
                
                # x1(t)
                ax1.plot(time, traj[:, 0], color=color, linewidth=2, 
                        label=f"CI: {result['initial']}")
                
                # x2(t)
                ax2.plot(time, traj[:, 1], color=color, linewidth=2, 
                        label=f"CI: {result['initial']}")
            
            ax1.set_ylabel('x₁(t)')
            ax1.set_title('Évolution Temporelle des États')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel('Temps (s)')
            ax2.set_ylabel('x₂(t)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du tracé: {str(e)}")
    
    def plot_lyapunov_function(self):
        """Tracer la fonction de Lyapunov"""
        if self.lyapunov_data is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord effectuer l'analyse de Lyapunov")
            return
        
        try:
            params = self.param_panel.get_parameters()
            x1_range = params.get('x1_range', 5.0)
            x2_range = params.get('x2_range', 5.0)
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Grille pour l'évaluation
            x1_grid = np.linspace(-x1_range, x1_range, 50)
            x2_grid = np.linspace(-x2_range, x2_range, 50)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            
            # Évaluation de la fonction de Lyapunov
            V = np.zeros_like(X1)
            lyap_func = self.lyapunov_data['function']
            
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    state = np.array([X1[i, j], X2[i, j]])
                    V[i, j] = lyap_func(state)
            
            # Tracé en contours
            contours = ax.contour(X1, X2, V, levels=20, colors='blue', alpha=0.6)
            ax.clabel(contours, inline=True, fontsize=8)
            
            # Tracé en couleurs
            im = ax.contourf(X1, X2, V, levels=50, cmap='viridis', alpha=0.4)
            self.plot_canvas.figure.colorbar(im, ax=ax, label='V(x)')
            
            # Point d'équilibre
            if self.equilibrium_point is not None:
                ax.plot(self.equilibrium_point[0], self.equilibrium_point[1], 
                       'ro', markersize=10, label='Équilibre')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('Fonction de Lyapunov V(x)')
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du tracé: {str(e)}")
    
    def plot_comparison(self):
        """Comparer système linéarisé vs non-linéaire"""
        if self.linearized_system is None or not hasattr(self, 'simulation_data'):
            QMessageBox.warning(self, "Erreur", 
                              "Veuillez d'abord linéariser le système et effectuer une simulation")
            return
        
        try:
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Simuler le système linéarisé pour comparaison
            A = self.linearized_system['A']
            
            # Prendre la première condition initiale
            x0 = self.simulation_data['results'][0]['initial']
            x0_eq = np.array(x0) - self.equilibrium_point  # Relative au point d'équilibre
            
            params = self.param_panel.get_parameters()
            t_final = params.get('t_final', 20.0)
            dt = params.get('dt', 0.01)
            t = np.arange(0, t_final + dt, dt)
            
            # Simulation linéaire
            from scipy.linalg import expm
            x_linear = []
            for ti in t:
                x_lin = expm(A * ti) @ x0_eq + self.equilibrium_point
                x_linear.append(x_lin)
            
            x_linear = np.array(x_linear)
            
            # Trajectoire non-linéaire
            nonlinear_traj = self.simulation_data['results'][0]['trajectory']
            nonlinear_time = self.simulation_data['results'][0]['time']
            
            # Comparaison dans le plan de phase
            ax.plot(x_linear[:, 0], x_linear[:, 1], 'b--', linewidth=2, 
                   label='Système linéarisé')
            ax.plot(nonlinear_traj[:, 0], nonlinear_traj[:, 1], 'r-', linewidth=2, 
                   label='Système non-linéaire')
            
            # Points initiaux
            ax.plot(x0[0], x0[1], 'go', markersize=8, label='Condition initiale')
            
            # Point d'équilibre
            ax.plot(self.equilibrium_point[0], self.equilibrium_point[1], 
                   'ko', markersize=10, markerfacecolor='yellow', 
                   markeredgewidth=2, label='Équilibre')
            
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('Comparaison Linéaire vs Non-Linéaire')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du tracé: {str(e)}")
    
    def update_system(self, system_data: Dict[str, Any]):
        """Mise à jour du système depuis l'extérieur"""
        self.current_system.update(system_data)
    
    def export_data(self):
        """Exporter les données d'analyse non-linéaire"""
        # TODO: Implémenter l'export
        QMessageBox.information(self, "Export", "Export des données d'analyse non-linéaire à implémenter")
    
    def capture_figure(self):
        """Capturer la figure actuelle"""
        self.plot_canvas.save_figure()
