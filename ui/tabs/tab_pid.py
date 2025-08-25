#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet Régulateurs PID
Conception et réglage de régulateurs PID avec méthodes automatiques et manuelles
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTextEdit, QMessageBox, QLineEdit,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..widgets_common import MatrixEditor, PlotCanvas, ParameterPanel
from ...core.pid_design import (
    pid_ziegler_nichols, pid_cohen_coon, simulate_pid_control,
    calculate_step_response_parameters, pid_manual_tuning
)
from ...core.lti_tools import time_response, step_info
from ...core.spaces import matrices_to_tf
from ...core.utils import validate_system_matrices


class TabPID(QWidget):
    """Onglet de conception et réglage de régulateurs PID"""
    
    system_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.system_tf = None
        self.pid_gains = {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0}
        self.simulation_data = None
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
        
        # Éditeur de matrices / TF
        self.matrix_editor = MatrixEditor("Système à Réguler")
        left_layout.addWidget(self.matrix_editor)
        
        # Configuration PID
        self.setup_pid_configuration(left_layout)
        
        # Méthodes de réglage
        self.setup_tuning_methods(left_layout)
        
        # Paramètres de simulation
        self.param_panel = ParameterPanel("Paramètres de Simulation")
        self.param_panel.add_parameter("t_final", "Temps final", 20.0, 1.0, 100.0, 1, "s")
        self.param_panel.add_parameter("consigne", "Amplitude consigne", 1.0, 0.1, 10.0, 2)
        self.param_panel.add_parameter("t_consigne", "Temps consigne", 1.0, 0.0, 10.0, 2, "s")
        left_layout.addWidget(self.param_panel)
        
        # Zone de résultats textuels
        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        self.results_text.setMaximumHeight(250)
        left_layout.addWidget(self.results_text)
        
        main_splitter.addWidget(left_widget)
        
        # Zone de droite : visualisations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Canvas pour les graphiques
        self.plot_canvas = PlotCanvas("Réponse du Système avec PID")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de visualisation
        viz_buttons_layout = QHBoxLayout()
        
        viz_buttons = [
            ("Réponse PID", self.plot_pid_response),
            ("Comparaison Méthodes", self.plot_methods_comparison),
            ("Signal de Commande", self.plot_control_signal),
            ("Analyse Fréquentielle", self.plot_frequency_analysis)
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
        
    def setup_pid_configuration(self, parent_layout):
        """Configuration des paramètres PID"""
        pid_group = QGroupBox("Configuration du Régulateur PID")
        pid_layout = QVBoxLayout(pid_group)
        
        # Type de régulateur
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type de régulateur:"))
        
        self.combo_pid_type = QComboBox()
        self.combo_pid_type.addItems([
            "PID standard",
            "PI (sans dérivée)",
            "PD (sans intégrale)",
            "P (proportionnel seul)"
        ])
        type_layout.addWidget(self.combo_pid_type)
        type_layout.addStretch()
        pid_layout.addLayout(type_layout)
        
        # Réglage manuel des gains
        manual_group = QGroupBox("Réglage Manuel")
        manual_layout = QVBoxLayout(manual_group)
        
        # Gain proportionnel
        kp_layout = QHBoxLayout()
        kp_layout.addWidget(QLabel("Kp:"))
        self.spin_kp = QDoubleSpinBox()
        self.spin_kp.setRange(0.0, 1000.0)
        self.spin_kp.setValue(1.0)
        self.spin_kp.setDecimals(3)
        self.spin_kp.valueChanged.connect(self.on_pid_gains_changed)
        kp_layout.addWidget(self.spin_kp)
        
        self.slider_kp = QSlider(Qt.Horizontal)
        self.slider_kp.setRange(0, 1000)
        self.slider_kp.setValue(100)
        self.slider_kp.valueChanged.connect(lambda v: self.spin_kp.setValue(v/100))
        kp_layout.addWidget(self.slider_kp)
        manual_layout.addLayout(kp_layout)
        
        # Gain intégral
        ki_layout = QHBoxLayout()
        ki_layout.addWidget(QLabel("Ki:"))
        self.spin_ki = QDoubleSpinBox()
        self.spin_ki.setRange(0.0, 100.0)
        self.spin_ki.setValue(0.0)
        self.spin_ki.setDecimals(3)
        self.spin_ki.valueChanged.connect(self.on_pid_gains_changed)
        ki_layout.addWidget(self.spin_ki)
        
        self.slider_ki = QSlider(Qt.Horizontal)
        self.slider_ki.setRange(0, 1000)
        self.slider_ki.setValue(0)
        self.slider_ki.valueChanged.connect(lambda v: self.spin_ki.setValue(v/100))
        ki_layout.addWidget(self.slider_ki)
        manual_layout.addLayout(ki_layout)
        
        # Gain dérivé
        kd_layout = QHBoxLayout()
        kd_layout.addWidget(QLabel("Kd:"))
        self.spin_kd = QDoubleSpinBox()
        self.spin_kd.setRange(0.0, 10.0)
        self.spin_kd.setValue(0.0)
        self.spin_kd.setDecimals(3)
        self.spin_kd.valueChanged.connect(self.on_pid_gains_changed)
        kd_layout.addWidget(self.spin_kd)
        
        self.slider_kd = QSlider(Qt.Horizontal)
        self.slider_kd.setRange(0, 1000)
        self.slider_kd.setValue(0)
        self.slider_kd.valueChanged.connect(lambda v: self.spin_kd.setValue(v/100))
        kd_layout.addWidget(self.slider_kd)
        manual_layout.addLayout(kd_layout)
        
        pid_layout.addWidget(manual_group)
        
        # Anti-windup
        antiwindup_layout = QHBoxLayout()
        self.checkbox_antiwindup = QCheckBox("Anti-windup activé")
        self.checkbox_antiwindup.setChecked(True)
        antiwindup_layout.addWidget(self.checkbox_antiwindup)
        
        antiwindup_layout.addWidget(QLabel("Limite:"))
        self.spin_windup_limit = QDoubleSpinBox()
        self.spin_windup_limit.setRange(0.1, 1000.0)
        self.spin_windup_limit.setValue(10.0)
        self.spin_windup_limit.setDecimals(1)
        antiwindup_layout.addWidget(self.spin_windup_limit)
        
        antiwindup_layout.addStretch()
        pid_layout.addLayout(antiwindup_layout)
        
        parent_layout.addWidget(pid_group)
    
    def setup_tuning_methods(self, parent_layout):
        """Configuration des méthodes de réglage automatique"""
        tuning_group = QGroupBox("Méthodes de Réglage Automatique")
        tuning_layout = QVBoxLayout(tuning_group)
        
        # Boutons des méthodes
        methods_layout = QHBoxLayout()
        
        self.btn_ziegler_nichols = QPushButton("Ziegler-Nichols")
        self.btn_ziegler_nichols.setObjectName("actionButton")
        self.btn_ziegler_nichols.clicked.connect(self.apply_ziegler_nichols)
        methods_layout.addWidget(self.btn_ziegler_nichols)
        
        self.btn_cohen_coon = QPushButton("Cohen-Coon")
        self.btn_cohen_coon.setObjectName("actionButton")
        self.btn_cohen_coon.clicked.connect(self.apply_cohen_coon)
        methods_layout.addWidget(self.btn_cohen_coon)
        
        self.btn_manual_tuning = QPushButton("Réglage Optimisé")
        self.btn_manual_tuning.setObjectName("validateButton")
        self.btn_manual_tuning.clicked.connect(self.apply_optimized_tuning)
        methods_layout.addWidget(self.btn_manual_tuning)
        
        tuning_layout.addLayout(methods_layout)
        
        # Bouton de simulation
        self.btn_simulate = QPushButton("Simuler Régulation PID")
        self.btn_simulate.setObjectName("validateButton")
        self.btn_simulate.setMinimumHeight(40)
        self.btn_simulate.clicked.connect(self.simulate_pid_system)
        tuning_layout.addWidget(self.btn_simulate)
        
        parent_layout.addWidget(tuning_group)
    
    def setup_connections(self):
        """Configuration des connexions"""
        self.matrix_editor.matrix_changed.connect(self.on_matrices_changed)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Connexions des sliders avec les spinboxes
        self.spin_kp.valueChanged.connect(lambda v: self.slider_kp.setValue(int(v*100)))
        self.spin_ki.valueChanged.connect(lambda v: self.slider_ki.setValue(int(v*100)))
        self.spin_kd.valueChanged.connect(lambda v: self.slider_kd.setValue(int(v*100)))
    
    def on_matrices_changed(self, matrices: Dict[str, np.ndarray]):
        """Gestion du changement de matrices"""
        self.current_system.update(matrices)
        self.convert_to_transfer_function()
        self.clear_results()
    
    def on_parameters_changed(self, parameters: Dict[str, float]):
        """Gestion du changement de paramètres"""
        self.current_system.update(parameters)
    
    def on_pid_gains_changed(self):
        """Gestion du changement des gains PID"""
        self.pid_gains = {
            'Kp': self.spin_kp.value(),
            'Ki': self.spin_ki.value(),
            'Kd': self.spin_kd.value()
        }
        
        # Adaptation selon le type de régulateur
        pid_type = self.combo_pid_type.currentText()
        if "PI" in pid_type:
            self.pid_gains['Kd'] = 0.0
            self.spin_kd.setValue(0.0)
            self.spin_kd.setEnabled(False)
        elif "PD" in pid_type:
            self.pid_gains['Ki'] = 0.0
            self.spin_ki.setValue(0.0)
            self.spin_ki.setEnabled(False)
        elif "P " in pid_type:
            self.pid_gains['Ki'] = 0.0
            self.pid_gains['Kd'] = 0.0
            self.spin_ki.setValue(0.0)
            self.spin_kd.setValue(0.0)
            self.spin_ki.setEnabled(False)
            self.spin_kd.setEnabled(False)
        else:
            self.spin_ki.setEnabled(True)
            self.spin_kd.setEnabled(True)
    
    def convert_to_transfer_function(self):
        """Convertir les matrices en fonction de transfert"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            self.system_tf = None
            return
        
        try:
            # Validation du système
            is_valid, errors = validate_system_matrices(**matrices)
            if not is_valid:
                self.results_text.setText("❌ Système invalide:\n" + "\n".join(errors))
                return
            
            # Conversion en fonction de transfert
            self.system_tf = matrices_to_tf(**matrices)
            
            # Vérification SISO
            if matrices['B'].shape[1] > 1 or matrices['C'].shape[0] > 1:
                QMessageBox.warning(self, "Avertissement", 
                                  "Système MIMO détecté. Le réglage PID sera appliqué à la première E/S.")
            
            # Affichage des informations système
            step_params = calculate_step_response_parameters(self.system_tf)
            
            results = f"""
SYSTÈME À RÉGULER

Fonction de transfert:
{self.format_transfer_function(self.system_tf)}

Paramètres de la réponse indicielle:
• Gain statique: {step_params.get('static_gain', 'N/A'):.4f}
• Temps de montée: {step_params.get('rise_time', 'N/A'):.3f} s
• Temps d'établissement: {step_params.get('settling_time', 'N/A'):.3f} s
• Dépassement: {step_params.get('overshoot', 'N/A'):.1f} %
• Temps de pic: {step_params.get('peak_time', 'N/A'):.3f} s

Système prêt pour le réglage PID !
            """
            
            self.results_text.setText(results.strip())
            
        except Exception as e:
            error_msg = f"Erreur de conversion: {str(e)}"
            self.results_text.setText(f"❌ {error_msg}")
    
    def clear_results(self):
        """Effacer les résultats"""
        pass  # Les résultats sont mis à jour automatiquement
    
    def format_transfer_function(self, tf_sys) -> str:
        """Formatage de la fonction de transfert"""
        try:
            import control
            if hasattr(tf_sys, 'num') and hasattr(tf_sys, 'den'):
                num = tf_sys.num[0][0] if isinstance(tf_sys.num[0], list) else tf_sys.num[0]
                den = tf_sys.den[0][0] if isinstance(tf_sys.den[0], list) else tf_sys.den[0]
                
                # Formatage polynomial
                num_str = self.format_polynomial(num)
                den_str = self.format_polynomial(den)
                
                return f"G(s) = ({num_str}) / ({den_str})"
            else:
                return str(tf_sys)
        except:
            return "Format non disponible"
    
    def format_polynomial(self, coeffs: np.ndarray) -> str:
        """Formatage d'un polynôme"""
        if len(coeffs) == 0:
            return "0"
        
        terms = []
        n = len(coeffs) - 1
        
        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-10:
                continue
            
            power = n - i
            
            if power == 0:
                terms.append(f"{coeff:.4g}")
            elif power == 1:
                if coeff == 1:
                    terms.append("s")
                elif coeff == -1:
                    terms.append("-s")
                else:
                    terms.append(f"{coeff:.4g}s")
            else:
                if coeff == 1:
                    terms.append(f"s^{power}")
                elif coeff == -1:
                    terms.append(f"-s^{power}")
                else:
                    terms.append(f"{coeff:.4g}s^{power}")
        
        if not terms:
            return "0"
        
        result = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                result += f" {term}"
            else:
                result += f" + {term}"
        
        return result
    
    def apply_ziegler_nichols(self):
        """Appliquer la méthode de Ziegler-Nichols"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            Kp, Ki, Kd = pid_ziegler_nichols(self.system_tf)
            
            # Mettre à jour les gains
            self.spin_kp.setValue(Kp)
            self.spin_ki.setValue(Ki)
            self.spin_kd.setValue(Kd)
            
            self.pid_gains = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            
            # Affichage des résultats
            results = f"""
MÉTHODE ZIEGLER-NICHOLS APPLIQUÉE

Gains calculés:
• Kp = {Kp:.4f}
• Ki = {Ki:.4f}
• Kd = {Kd:.4f}

Fonction de transfert du régulateur:
C(s) = {Kp:.4f} + {Ki:.4f}/s + {Kd:.4f}s

Note: Ces gains sont basés sur la réponse indicielle du système.
Vous pouvez les ajuster manuellement si nécessaire.
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Ziegler-Nichols", 
                                  "Gains PID calculés selon la méthode de Ziegler-Nichols !")
            
        except Exception as e:
            error_msg = f"Erreur lors du calcul Ziegler-Nichols: {str(e)}"
            QMessageBox.warning(self, "Erreur", error_msg)
            self.results_text.setText(f"❌ {error_msg}")
    
    def apply_cohen_coon(self):
        """Appliquer la méthode de Cohen-Coon"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            Kp, Ki, Kd = pid_cohen_coon(self.system_tf)
            
            # Mettre à jour les gains
            self.spin_kp.setValue(Kp)
            self.spin_ki.setValue(Ki)
            self.spin_kd.setValue(Kd)
            
            self.pid_gains = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            
            # Affichage des résultats
            results = f"""
MÉTHODE COHEN-COON APPLIQUÉE

Gains calculés:
• Kp = {Kp:.4f}
• Ki = {Ki:.4f}
• Kd = {Kd:.4f}

Fonction de transfert du régulateur:
C(s) = {Kp:.4f} + {Ki:.4f}/s + {Kd:.4f}s

Note: Cette méthode est optimisée pour les systèmes avec retard.
Ajustement manuel possible selon les performances désirées.
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Cohen-Coon", 
                                  "Gains PID calculés selon la méthode de Cohen-Coon !")
            
        except Exception as e:
            error_msg = f"Erreur lors du calcul Cohen-Coon: {str(e)}"
            QMessageBox.warning(self, "Erreur", error_msg)
            self.results_text.setText(f"❌ {error_msg}")
    
    def apply_optimized_tuning(self):
        """Appliquer un réglage optimisé personnalisé"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            # Spécifications par défaut
            specs = {
                'overshoot_max': 10.0,      # % maximum
                'settling_time_max': 5.0,    # secondes
                'rise_time_max': 2.0         # secondes
            }
            
            Kp, Ki, Kd = pid_manual_tuning(self.system_tf, specs)
            
            # Mettre à jour les gains
            self.spin_kp.setValue(Kp)
            self.spin_ki.setValue(Ki)
            self.spin_kd.setValue(Kd)
            
            self.pid_gains = {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
            
            # Affichage des résultats
            results = f"""
RÉGLAGE OPTIMISÉ APPLIQUÉ

Spécifications cibles:
• Dépassement max: {specs['overshoot_max']:.1f} %
• Temps d'établissement max: {specs['settling_time_max']:.1f} s
• Temps de montée max: {specs['rise_time_max']:.1f} s

Gains optimisés:
• Kp = {Kp:.4f}
• Ki = {Ki:.4f}
• Kd = {Kd:.4f}

Ce réglage vise un compromis performance/robustesse.
            """
            
            self.results_text.setText(results.strip())
            
            QMessageBox.information(self, "Réglage Optimisé", 
                                  "Gains PID optimisés calculés !")
            
        except Exception as e:
            error_msg = f"Erreur lors du réglage optimisé: {str(e)}"
            QMessageBox.warning(self, "Erreur", error_msg)
            self.results_text.setText(f"❌ {error_msg}")
    
    def simulate_pid_system(self):
        """Simuler le système avec régulation PID"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            # Paramètres de simulation
            params = self.param_panel.get_parameters()
            t_final = params.get('t_final', 20.0)
            consigne_amp = params.get('consigne', 1.0)
            t_consigne = params.get('t_consigne', 1.0)
            
            # Configuration anti-windup
            antiwindup_config = {
                'enabled': self.checkbox_antiwindup.isChecked(),
                'limit': self.spin_windup_limit.value()
            }
            
            # Simulation
            simulation_result = simulate_pid_control(
                self.system_tf, 
                self.pid_gains['Kp'], 
                self.pid_gains['Ki'], 
                self.pid_gains['Kd'],
                t_final=t_final,
                setpoint_amplitude=consigne_amp,
                setpoint_time=t_consigne,
                antiwindup=antiwindup_config
            )
            
            if simulation_result['success']:
                self.simulation_data = simulation_result
                
                # Calcul des performances
                performance = self.calculate_performance_metrics(simulation_result)
                
                # Affichage des résultats
                results = f"""
SIMULATION PID TERMINÉE

Configuration:
• Kp = {self.pid_gains['Kp']:.4f}
• Ki = {self.pid_gains['Ki']:.4f}
• Kd = {self.pid_gains['Kd']:.4f}
• Anti-windup: {'Activé' if antiwindup_config['enabled'] else 'Désactivé'}

Performances obtenues:
• Temps de montée: {performance['rise_time']:.3f} s
• Dépassement: {performance['overshoot']:.1f} %
• Temps d'établissement: {performance['settling_time']:.3f} s
• Erreur statique: {performance['steady_state_error']:.4f}
• ISE (Intégrale erreur²): {performance['ise']:.6f}

✅ Simulation réussie !
                """
                
                self.results_text.setText(results.strip())
                
                # Tracer automatiquement
                self.plot_pid_response()
                
                QMessageBox.information(self, "Simulation", "Simulation PID terminée avec succès !")
                
            else:
                error_msg = simulation_result.get('error', 'Erreur inconnue')
                self.results_text.setText(f"❌ Échec simulation: {error_msg}")
                QMessageBox.warning(self, "Erreur", f"Simulation échouée: {error_msg}")
            
        except Exception as e:
            error_msg = f"Erreur lors de la simulation: {str(e)}"
            QMessageBox.warning(self, "Erreur", error_msg)
            self.results_text.setText(f"❌ {error_msg}")
    
    def calculate_performance_metrics(self, sim_data: Dict) -> Dict[str, float]:
        """Calculer les métriques de performance"""
        try:
            time = sim_data['time']
            output = sim_data['output']
            setpoint = sim_data['setpoint']
            error = setpoint - output
            
            # Temps de montée (10% à 90% de la valeur finale)
            final_value = np.mean(output[-100:])  # Moyenne des 100 derniers points
            idx_10 = np.where(output >= 0.1 * final_value)[0]
            idx_90 = np.where(output >= 0.9 * final_value)[0]
            
            rise_time = time[idx_90[0]] - time[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else 0.0
            
            # Dépassement
            max_output = np.max(output)
            overshoot = max(0, (max_output - final_value) / final_value * 100)
            
            # Temps d'établissement (±2% de la valeur finale)
            settling_indices = np.where(np.abs(output - final_value) <= 0.02 * abs(final_value))[0]
            settling_time = time[settling_indices[0]] if len(settling_indices) > 0 else time[-1]
            
            # Erreur statique
            steady_state_error = abs(np.mean(error[-100:]))
            
            # ISE (Integral of Squared Error)
            ise = np.trapz(error**2, time)
            
            return {
                'rise_time': rise_time,
                'overshoot': overshoot,
                'settling_time': settling_time,
                'steady_state_error': steady_state_error,
                'ise': ise
            }
            
        except Exception as e:
            return {
                'rise_time': 0.0,
                'overshoot': 0.0,
                'settling_time': 0.0,
                'steady_state_error': 0.0,
                'ise': float('inf')
            }
    
    def plot_pid_response(self):
        """Tracer la réponse du système avec PID"""
        if self.simulation_data is None:
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Tracé de la consigne et de la sortie
            ax.plot(data['time'], data['setpoint'], 'k--', linewidth=2, label='Consigne')
            ax.plot(data['time'], data['output'], 'b-', linewidth=2, label='Sortie')
            
            # Zone de tolérance ±2%
            final_value = np.mean(data['output'][-100:])
            tolerance = 0.02 * abs(final_value)
            ax.fill_between(data['time'], 
                           final_value - tolerance, 
                           final_value + tolerance, 
                           alpha=0.2, color='green', label='Tolérance ±2%')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Réponse PID (Kp={self.pid_gains["Kp"]:.3f}, Ki={self.pid_gains["Ki"]:.3f}, Kd={self.pid_gains["Kd"]:.3f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_methods_comparison(self):
        """Comparer les différentes méthodes de réglage"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            # Calculer les gains pour chaque méthode
            methods = {}
            
            try:
                Kp_zn, Ki_zn, Kd_zn = pid_ziegler_nichols(self.system_tf)
                methods['Ziegler-Nichols'] = {'Kp': Kp_zn, 'Ki': Ki_zn, 'Kd': Kd_zn, 'color': 'blue'}
            except:
                pass
            
            try:
                Kp_cc, Ki_cc, Kd_cc = pid_cohen_coon(self.system_tf)
                methods['Cohen-Coon'] = {'Kp': Kp_cc, 'Ki': Ki_cc, 'Kd': Kd_cc, 'color': 'green'}
            except:
                pass
            
            # Ajouter la méthode manuelle actuelle
            methods['Manuel'] = {
                'Kp': self.pid_gains['Kp'], 
                'Ki': self.pid_gains['Ki'], 
                'Kd': self.pid_gains['Kd'],
                'color': 'red'
            }
            
            if len(methods) < 2:
                QMessageBox.warning(self, "Erreur", "Pas assez de méthodes disponibles pour comparaison")
                return
            
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Paramètres de simulation
            params = self.param_panel.get_parameters()
            t_final = params.get('t_final', 20.0)
            
            # Simuler chaque méthode
            for method_name, gains in methods.items():
                try:
                    sim_result = simulate_pid_control(
                        self.system_tf, 
                        gains['Kp'], gains['Ki'], gains['Kd'],
                        t_final=t_final
                    )
                    
                    if sim_result['success']:
                        ax.plot(sim_result['time'], sim_result['output'], 
                               color=gains['color'], linewidth=2, label=method_name)
                except:
                    continue
            
            # Consigne
            t = np.linspace(0, t_final, 1000)
            setpoint = np.ones_like(t)
            ax.plot(t, setpoint, 'k--', linewidth=2, label='Consigne')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Sortie')
            ax.set_title('Comparaison des Méthodes de Réglage PID')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de comparaison: {str(e)}")
    
    def plot_control_signal(self):
        """Tracer le signal de commande"""
        if self.simulation_data is None:
            QMessageBox.warning(self, "Erreur", "Aucune donnée de simulation disponible")
            return
        
        try:
            data = self.simulation_data
            
            self.plot_canvas.clear_plot()
            
            # Deux sous-graphiques
            ax1 = self.plot_canvas.figure.add_subplot(2, 1, 1)
            ax2 = self.plot_canvas.figure.add_subplot(2, 1, 2)
            
            # Graphique 1: Sortie
            ax1.plot(data['time'], data['setpoint'], 'k--', linewidth=2, label='Consigne')
            ax1.plot(data['time'], data['output'], 'b-', linewidth=2, label='Sortie')
            ax1.set_ylabel('Sortie')
            ax1.set_title('Réponse du Système')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Graphique 2: Signal de commande
            ax2.plot(data['time'], data['control'], 'r-', linewidth=2, label='Commande u(t)')
            
            # Limites anti-windup si activé
            if self.checkbox_antiwindup.isChecked():
                limit = self.spin_windup_limit.value()
                ax2.axhline(y=limit, color='red', linestyle='--', alpha=0.5, label=f'Limite +{limit}')
                ax2.axhline(y=-limit, color='red', linestyle='--', alpha=0.5, label=f'Limite -{limit}')
            
            ax2.set_xlabel('Temps (s)')
            ax2.set_ylabel('Signal de Commande')
            ax2.set_title('Signal de Commande PID')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de tracé: {str(e)}")
    
    def plot_frequency_analysis(self):
        """Tracer l'analyse fréquentielle du système bouclé"""
        if self.system_tf is None:
            QMessageBox.warning(self, "Erreur", "Veuillez d'abord définir un système valide")
            return
        
        try:
            import control
            
            # Fonction de transfert du PID
            Kp, Ki, Kd = self.pid_gains['Kp'], self.pid_gains['Ki'], self.pid_gains['Kd']
            
            # C(s) = Kp + Ki/s + Kd*s
            if Ki != 0 and Kd != 0:
                # PID complet
                num_pid = [Kd, Kp, Ki]
                den_pid = [1, 0]
            elif Ki != 0:
                # PI
                num_pid = [Kp, Ki]
                den_pid = [1, 0]
            elif Kd != 0:
                # PD
                num_pid = [Kd, Kp]
                den_pid = [1]
            else:
                # P
                num_pid = [Kp]
                den_pid = [1]
            
            pid_tf = control.tf(num_pid, den_pid)
            
            # Fonction de transfert en boucle fermée
            loop_tf = control.series(pid_tf, self.system_tf)
            closed_loop_tf = control.feedback(loop_tf, 1)
            
            # Calcul de la réponse en fréquence
            w = np.logspace(-2, 2, 1000)
            mag_ol, phase_ol, omega_ol = control.bode_plot(loop_tf, w, plot=False)
            mag_cl, phase_cl, omega_cl = control.bode_plot(closed_loop_tf, w, plot=False)
            
            self.plot_canvas.clear_plot()
            
            # Quatre sous-graphiques
            ax1 = self.plot_canvas.figure.add_subplot(2, 2, 1)
            ax2 = self.plot_canvas.figure.add_subplot(2, 2, 2)
            ax3 = self.plot_canvas.figure.add_subplot(2, 2, 3)
            ax4 = self.plot_canvas.figure.add_subplot(2, 2, 4)
            
            # Bode boucle ouverte
            ax1.semilogx(omega_ol, 20*np.log10(mag_ol), 'b-', linewidth=2)
            ax1.set_ylabel('Module (dB)')
            ax1.set_title('Bode Boucle Ouverte')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            ax2.semilogx(omega_ol, phase_ol * 180/np.pi, 'b-', linewidth=2)
            ax2.set_ylabel('Phase (°)')
            ax2.set_xlabel('Fréquence (rad/s)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=-180, color='red', linestyle='--', alpha=0.5)
            
            # Bode boucle fermée
            ax3.semilogx(omega_cl, 20*np.log10(mag_cl), 'r-', linewidth=2)
            ax3.set_ylabel('Module (dB)')
            ax3.set_title('Bode Boucle Fermée')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=-3, color='orange', linestyle='--', alpha=0.5, label='-3dB')
            
            ax4.semilogx(omega_cl, phase_cl * 180/np.pi, 'r-', linewidth=2)
            ax4.set_ylabel('Phase (°)')
            ax4.set_xlabel('Fréquence (rad/s)')
            ax4.grid(True, alpha=0.3)
            
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur d'analyse fréquentielle: {str(e)}")
    
    def update_system(self, system_data: Dict[str, Any]):
        """Mise à jour du système depuis l'extérieur"""
        self.current_system.update(system_data)
        if 'A' in system_data:
            self.matrix_editor.set_matrices(system_data)
    
    def export_data(self):
        """Exporter les données PID"""
        # TODO: Implémenter l'export
        QMessageBox.information(self, "Export", "Export des données PID à implémenter")
    
    def capture_figure(self):
        """Capturer la figure actuelle"""
        self.plot_canvas.save_figure()
