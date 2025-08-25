#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Onglet d'analyse d'état
Analyse des systèmes linéaires : pôles, zéros, réponses temporelles et fréquentielles
"""

import numpy as np
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QLabel, QPushButton, QSplitter, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..widgets_common import MatrixEditor, PlotCanvas, ParameterPanel
from ...core.lti_tools import poles, zeros, time_response, bode_data
from ...core.spaces import matrices_to_tf, tf_to_matrices
from ...core.utils import validate_system_matrices


class TabStateAnalysis(QWidget):
    """Onglet d'analyse d'état des systèmes linéaires"""
    
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
        self.matrix_editor = MatrixEditor("Définition du Système")
        left_layout.addWidget(self.matrix_editor)
        
        # Paramètres de simulation
        self.param_panel = ParameterPanel("Paramètres de Simulation")
        self.param_panel.add_parameter("t_final", "Temps final (s)", 10.0, 0.1, 100.0)
        self.param_panel.add_parameter("dt", "Pas d'échantillonnage", 0.01, 0.001, 1.0, 3)
        left_layout.addWidget(self.param_panel)
        
        # Type d'entrée
        input_group = QGroupBox("Type d'Entrée")
        input_layout = QVBoxLayout(input_group)
        
        self.input_combo = QComboBox()
        self.input_combo.addItems([
            "Échelon unitaire",
            "Impulsion",
            "Sinusoïde",
            "Rampe",
            "Entrée personnalisée"
        ])
        input_layout.addWidget(self.input_combo)
        
        # Boutons d'analyse
        self.btn_analyze = QPushButton("Analyser le Système")
        self.btn_analyze.setObjectName("actionButton")
        self.btn_analyze.clicked.connect(self.analyze_system)
        input_layout.addWidget(self.btn_analyze)
        
        left_layout.addWidget(input_group)
        
        # Zone de résultats textuels
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setObjectName("resultsText")
        left_layout.addWidget(self.results_text)
        
        left_layout.addStretch()
        main_splitter.addWidget(left_widget)
        
        # Zone de droite : graphiques
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Canvas pour les graphiques
        self.plot_canvas = PlotCanvas("Analyse du Système")
        right_layout.addWidget(self.plot_canvas)
        
        # Boutons de graphiques
        plot_buttons_layout = QHBoxLayout()
        
        plot_buttons = [
            ("Réponse Temporelle", self.plot_time_response),
            ("Diagramme de Bode", self.plot_bode),
            ("Pôles/Zéros", self.plot_poles_zeros),
            ("Réponse Impulsionnelle", self.plot_impulse_response)
        ]
        
        for text, callback in plot_buttons:
            btn = QPushButton(text)
            btn.setObjectName("plotButton")
            btn.clicked.connect(callback)
            plot_buttons_layout.addWidget(btn)
        
        right_layout.addLayout(plot_buttons_layout)
        main_splitter.addWidget(right_widget)
        
        # Proportions du splitter
        main_splitter.setSizes([400, 800])
        
    def setup_connections(self):
        """Configuration des connexions"""
        self.matrix_editor.matrix_changed.connect(self.on_matrices_changed)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)
        self.input_combo.currentTextChanged.connect(self.on_input_changed)
    
    def on_matrices_changed(self, matrices: Dict[str, np.ndarray]):
        """Gestion du changement de matrices"""
        self.current_system.update(matrices)
        self.update_system_info()
    
    def on_parameters_changed(self, parameters: Dict[str, float]):
        """Gestion du changement de paramètres"""
        self.current_system.update(parameters)
    
    def on_input_changed(self, input_type: str):
        """Gestion du changement de type d'entrée"""
        self.current_system['input_type'] = input_type
    
    def update_system_info(self):
        """Mise à jour des informations système"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            self.results_text.setText("Système incomplet. Veuillez définir toutes les matrices.")
            return
        
        try:
            # Validation du système
            is_valid, errors = validate_system_matrices(**matrices)
            
            if not is_valid:
                self.results_text.setText(f"Erreurs de validation:\n" + "\n".join(errors))
                return
            
            # Calcul des propriétés
            A = matrices['A']
            system_poles = poles(A)
            system_zeros = zeros(matrices['A'], matrices['B'], matrices['C'], matrices['D'])
            
            # Stabilité
            is_stable = np.all(np.real(system_poles) < 0)
            stability_text = "STABLE" if is_stable else "INSTABLE"
            
            # Affichage des résultats
            results = f"""
ANALYSE DU SYSTÈME

Dimensions:
- États: {A.shape[0]}
- Entrées: {matrices['B'].shape[1]}
- Sorties: {matrices['C'].shape[0]}

Pôles du système:
{self.format_complex_array(system_poles)}

Zéros du système:
{self.format_complex_array(system_zeros)}

Stabilité: {stability_text}

Type de système: {'SISO' if matrices['B'].shape[1] == 1 and matrices['C'].shape[0] == 1 else 'MIMO'}
            """
            
            self.results_text.setText(results.strip())
            
        except Exception as e:
            self.results_text.setText(f"Erreur d'analyse: {str(e)}")
    
    def format_complex_array(self, arr: np.ndarray) -> str:
        """Formatage d'un tableau de nombres complexes"""
        if len(arr) == 0:
            return "Aucun"
        
        formatted = []
        for val in arr:
            if np.isreal(val):
                formatted.append(f"{val.real:.4f}")
            else:
                sign = '+' if val.imag >= 0 else '-'
                formatted.append(f"{val.real:.4f} {sign} {abs(val.imag):.4f}j")
        
        return "\n".join(formatted)
    
    def analyze_system(self):
        """Analyse complète du système"""
        try:
            self.update_system_info()
            self.plot_time_response()
            QMessageBox.information(self, "Analyse", "Analyse terminée avec succès")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'analyse: {str(e)}")
    
    def plot_time_response(self):
        """Tracer la réponse temporelle"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices système incomplètes")
            return
        
        try:
            # Paramètres de simulation
            t_final = self.current_system.get('t_final', 10.0)
            dt = self.current_system.get('dt', 0.01)
            input_type = self.current_system.get('input_type', 'Échelon unitaire')
            
            # Calcul de la réponse
            response_data = time_response(
                matrices['A'], matrices['B'], matrices['C'], matrices['D'],
                input_type, t_final, dt
            )
            
            # Traçage
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            t = response_data['time']
            y = response_data['output']
            u = response_data['input']
            
            # Graphique de sortie
            if y.ndim == 1:
                ax.plot(t, y, 'b-', linewidth=2, label='Sortie')
            else:
                for i in range(y.shape[1]):
                    ax.plot(t, y[:, i], linewidth=2, label=f'Sortie {i+1}')
            
            # Graphique d'entrée (en rouge)
            ax_twin = ax.twinx()
            if u.ndim == 1:
                ax_twin.plot(t, u, 'r--', linewidth=1.5, alpha=0.7, label='Entrée')
            else:
                for i in range(u.shape[1]):
                    ax_twin.plot(t, u[:, i], '--', linewidth=1.5, alpha=0.7, label=f'Entrée {i+1}')
            
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Sortie', color='blue')
            ax_twin.set_ylabel('Entrée', color='red')
            ax.tick_params(axis='y', labelcolor='blue')
            ax_twin.tick_params(axis='y', labelcolor='red')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            ax.set_title(f'Réponse Temporelle - {input_type}')
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur de tracé", f"Erreur lors du tracé: {str(e)}")
    
    def plot_bode(self):
        """Tracer le diagramme de Bode"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices système incomplètes")
            return
        
        try:
            # Conversion en fonction de transfert
            sys_tf = matrices_to_tf(**matrices)
            
            # Calcul des données de Bode
            w = np.logspace(-2, 2, 1000)
            bode_result = bode_data(sys_tf, w)
            
            # Traçage
            self.plot_canvas.clear_plot()
            
            # Module
            ax1 = self.plot_canvas.figure.add_subplot(2, 1, 1)
            ax1.semilogx(bode_result['frequency'], bode_result['magnitude'], 'b-', linewidth=2)
            ax1.set_ylabel('Module (dB)', color='blue')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Diagramme de Bode')
            
            # Phase
            ax2 = self.plot_canvas.figure.add_subplot(2, 1, 2)
            ax2.semilogx(bode_result['frequency'], bode_result['phase'], 'r-', linewidth=2)
            ax2.set_xlabel('Fréquence (rad/s)')
            ax2.set_ylabel('Phase (°)', color='red')
            ax2.grid(True, alpha=0.3)
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur de tracé", f"Erreur lors du tracé de Bode: {str(e)}")
    
    def plot_poles_zeros(self):
        """Tracer les pôles et zéros"""
        matrices = {k: v for k, v in self.current_system.items() if k in ['A', 'B', 'C', 'D']}
        
        if not all(v is not None for v in matrices.values()):
            QMessageBox.warning(self, "Erreur", "Matrices système incomplètes")
            return
        
        try:
            A = matrices['A']
            system_poles = poles(A)
            system_zeros = zeros(matrices['A'], matrices['B'], matrices['C'], matrices['D'])
            
            # Traçage
            self.plot_canvas.clear_plot()
            ax = self.plot_canvas.get_axes()
            
            # Pôles (croix bleues)
            if len(system_poles) > 0:
                ax.scatter(np.real(system_poles), np.imag(system_poles), 
                          marker='x', s=100, c='blue', linewidth=3, label='Pôles')
            
            # Zéros (cercles rouges)
            if len(system_zeros) > 0:
                ax.scatter(np.real(system_zeros), np.imag(system_zeros), 
                          marker='o', s=100, facecolors='none', edgecolors='red', 
                          linewidth=2, label='Zéros')
            
            # Cercle unité et axes
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Cercle unité')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('Partie Réelle')
            ax.set_ylabel('Partie Imaginaire')
            ax.set_title('Pôles et Zéros du Système')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
            
            self.plot_canvas.refresh()
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur de tracé", f"Erreur lors du tracé pôles/zéros: {str(e)}")
    
    def plot_impulse_response(self):
        """Tracer la réponse impulsionnelle"""
        # Sauvegarder le type d'entrée actuel
        current_input = self.current_system.get('input_type', 'Échelon unitaire')
        
        # Changer temporairement pour impulsion
        self.current_system['input_type'] = 'Impulsion'
        
        # Tracer la réponse
        self.plot_time_response()
        
        # Restaurer le type d'entrée
        self.current_system['input_type'] = current_input
    
    def update_system(self, system_data: Dict[str, Any]):
        """Mise à jour du système depuis l'extérieur"""
        self.current_system.update(system_data)
        if 'A' in system_data:
            self.matrix_editor.set_matrices(system_data)
    
    def export_data(self):
        """Exporter les données de l'onglet"""
        # TODO: Implémenter l'export des données d'analyse
        QMessageBox.information(self, "Export", "Export des données d'analyse à implémenter")
    
    def capture_figure(self):
        """Capturer la figure actuelle"""
        self.plot_canvas.save_figure()
