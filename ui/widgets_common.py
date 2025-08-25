#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Widgets communs pour ControlSysLab
Éditeurs de matrices, canvas matplotlib, sélecteurs, etc.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QGroupBox, QFrame, QTextEdit,
    QSplitter, QSizePolicy, QHeaderView, QMessageBox, QFileDialog,
    QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd


class MatrixEditor(QGroupBox):
    """Éditeur de matrices A, B, C, D avec interface améliorée et plus grande"""
    
    matrix_changed = pyqtSignal(dict)
    
    def __init__(self, title: str = "Matrices du Système"):
        super().__init__(title)
        self.matrices = {'A': None, 'B': None, 'C': None, 'D': None}
        self.tables = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur améliorée"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Zone de dimensions avec style amélioré
        dims_frame = QFrame()
        dims_frame.setObjectName("dimensionsFrame")
        dims_layout = QHBoxLayout(dims_frame)
        dims_layout.setSpacing(15)
        
        # Dimensions avec labels plus clairs
        dims_layout.addWidget(QLabel("Nombre d'états (n):"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 10)
        self.spin_n.setValue(2)
        self.spin_n.setMinimumWidth(80)
        self.spin_n.valueChanged.connect(self.update_dimensions)
        dims_layout.addWidget(self.spin_n)
        
        dims_layout.addWidget(QLabel("Entrées (m):"))
        self.spin_m = QSpinBox()
        self.spin_m.setRange(1, 5)
        self.spin_m.setValue(1)
        self.spin_m.setMinimumWidth(80)
        self.spin_m.valueChanged.connect(self.update_dimensions)
        dims_layout.addWidget(self.spin_m)
        
        dims_layout.addWidget(QLabel("Sorties (p):"))
        self.spin_p = QSpinBox()
        self.spin_p.setRange(1, 5)
        self.spin_p.setValue(1)
        self.spin_p.setMinimumWidth(80)
        self.spin_p.valueChanged.connect(self.update_dimensions)
        dims_layout.addWidget(self.spin_p)
        
        dims_layout.addStretch()
        layout.addWidget(dims_frame)
        
        # Zone des matrices avec scroll et taille améliorée
        matrices_scroll = QScrollArea()
        matrices_scroll.setWidgetResizable(True)
        matrices_scroll.setMinimumHeight(400)
        
        matrices_widget = QWidget()
        matrices_layout = QGridLayout(matrices_widget)
        matrices_layout.setSpacing(20)
        
        # Création des tables pour chaque matrice avec taille augmentée
        matrix_info = {
            'A': ("A (n×n) - Matrice d'État", 0, 0),
            'B': ("B (n×m) - Matrice d'Entrée", 0, 1),
            'C': ("C (p×n) - Matrice de Sortie", 1, 0),
            'D': ("D (p×m) - Transmission Directe", 1, 1)
        }
        
        for matrix_name, (label, row, col) in matrix_info.items():
            group = QGroupBox(label)
            group.setObjectName("matrixGroup")
            group_layout = QVBoxLayout(group)
            
            table = QTableWidget()
            table.setObjectName("matrixTable")
            table.setMinimumSize(300, 200)  # Taille augmentée
            
            # Configuration pour des cellules beaucoup plus grandes et confortables
            table.verticalHeader().setDefaultSectionSize(45)  # Augmenté de 35 à 45
            table.horizontalHeader().setDefaultSectionSize(100)  # Augmenté de 80 à 100
            table.verticalHeader().setVisible(True)
            table.horizontalHeader().setVisible(True)
            
            # Police plus grande et plus lisible
            font = QFont()
            font.setPointSize(12)  # Augmenté de 11 à 12
            font.setBold(False)
            table.setFont(font)
            
            # Style des headers
            table.horizontalHeader().setStyleSheet("""
                QHeaderView::section {
                    background-color: #0D47A1;
                    color: white;
                    font-weight: bold;
                    border: 1px solid #ccc;
                    padding: 8px;
                }
            """)
            
            table.verticalHeader().setStyleSheet("""
                QHeaderView::section {
                    background-color: #0D47A1;
                    color: white;
                    font-weight: bold;
                    border: 1px solid #ccc;
                    padding: 8px;
                }
            """)
            
            # Connexion pour changements
            table.cellChanged.connect(lambda r, c, name=matrix_name: self.on_cell_changed(name, r, c))
            
            self.tables[matrix_name] = table
            group_layout.addWidget(table)
            
            matrices_layout.addWidget(group, row, col)
        
        matrices_scroll.setWidget(matrices_widget)
        layout.addWidget(matrices_scroll)
        
        # Boutons d'action avec couleurs améliorées
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        btn_random = QPushButton("🎲 Générer Aléatoirement")
        btn_random.setObjectName("actionButton")
        btn_random.setMinimumHeight(40)
        btn_random.clicked.connect(self.generate_random)
        buttons_layout.addWidget(btn_random)
        
        btn_examples = QPushButton("📚 Exemples")
        btn_examples.setObjectName("actionButton")
        btn_examples.setMinimumHeight(40)
        btn_examples.clicked.connect(self.load_examples)
        buttons_layout.addWidget(btn_examples)
        
        btn_import = QPushButton("📁 Importer CSV")
        btn_import.setObjectName("actionButton")
        btn_import.setMinimumHeight(40)
        btn_import.clicked.connect(self.import_csv)
        buttons_layout.addWidget(btn_import)
        
        btn_clear = QPushButton("🗑️ Effacer")
        btn_clear.setObjectName("errorButton")
        btn_clear.setMinimumHeight(40)
        btn_clear.clicked.connect(self.clear_matrices)
        buttons_layout.addWidget(btn_clear)
        
        btn_validate = QPushButton("✅ Valider")
        btn_validate.setObjectName("validateButton")
        btn_validate.setMinimumHeight(40)
        btn_validate.clicked.connect(self.validate_matrices)
        buttons_layout.addWidget(btn_validate)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Initialiser les dimensions
        self.update_dimensions()
        
    def update_dimensions(self):
        """Mise à jour des dimensions des matrices"""
        n = self.spin_n.value()
        m = self.spin_m.value()
        p = self.spin_p.value()
        
        dimensions = {
            'A': (n, n),
            'B': (n, m),
            'C': (p, n),
            'D': (p, m)
        }
        
        for matrix_name, (rows, cols) in dimensions.items():
            table = self.tables[matrix_name]
            table.setRowCount(rows)
            table.setColumnCount(cols)
            
            # Headers avec noms explicites
            if matrix_name == 'A':
                table.setHorizontalHeaderLabels([f"x{j+1}" for j in range(cols)])
                table.setVerticalHeaderLabels([f"ẋ{i+1}" for i in range(rows)])
            elif matrix_name == 'B':
                table.setHorizontalHeaderLabels([f"u{j+1}" for j in range(cols)])
                table.setVerticalHeaderLabels([f"ẋ{i+1}" for i in range(rows)])
            elif matrix_name == 'C':
                table.setHorizontalHeaderLabels([f"x{j+1}" for j in range(cols)])
                table.setVerticalHeaderLabels([f"y{i+1}" for i in range(rows)])
            elif matrix_name == 'D':
                table.setHorizontalHeaderLabels([f"u{j+1}" for j in range(cols)])
                table.setVerticalHeaderLabels([f"y{i+1}" for i in range(rows)])
            
            # Initialiser avec des zéros formatés
            for i in range(rows):
                for j in range(cols):
                    item = table.item(i, j)
                    if item is None:
                        item = QTableWidgetItem("0.0000")
                        item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(i, j, item)
                    
            # Redimensionnement automatique
            table.resizeColumnsToContents()
            table.resizeRowsToContents()
    
    def on_cell_changed(self, matrix_name: str, row: int, col: int):
        """Gestion du changement de valeur dans une cellule"""
        table = self.tables[matrix_name]
        item = table.item(row, col)
        if item:
            try:
                # Validation de la valeur numérique
                value = float(item.text())
                item.setText(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                
                # Style pour les valeurs valides
                item.setBackground(Qt.white)
                
            except ValueError:
                # Style pour les erreurs
                item.setBackground(Qt.red)
                item.setText("0.0000")
                item.setTextAlignment(Qt.AlignCenter)
        
        # Émettre le signal de changement après un délai
        QTimer.singleShot(500, self.emit_matrices_changed)
    
    def emit_matrices_changed(self):
        """Émettre le signal de changement des matrices"""
        self.matrices = self.get_matrices()
        self.matrix_changed.emit(self.matrices)
    
    def get_matrices(self) -> Dict[str, Optional[np.ndarray]]:
        """Récupérer les matrices sous forme numpy"""
        matrices = {}
        
        for matrix_name, table in self.tables.items():
            rows = table.rowCount()
            cols = table.columnCount()
            
            if rows == 0 or cols == 0:
                matrices[matrix_name] = None
                continue
                
            matrix = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    item = table.item(i, j)
                    if item:
                        try:
                            matrix[i, j] = float(item.text())
                        except ValueError:
                            matrix[i, j] = 0.0
            
            matrices[matrix_name] = matrix
        
        return matrices
    
    def set_matrices(self, matrices: Dict[str, np.ndarray]):
        """Définir les matrices"""
        for matrix_name, matrix in matrices.items():
            if matrix_name in self.tables and matrix is not None:
                table = self.tables[matrix_name]
                rows, cols = matrix.shape
                
                table.setRowCount(rows)
                table.setColumnCount(cols)
                
                for i in range(rows):
                    for j in range(cols):
                        item = QTableWidgetItem(f"{matrix[i, j]:.4f}")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setBackground(Qt.white)
                        table.setItem(i, j, item)
        
        self.emit_matrices_changed()
    
    def generate_random(self):
        """Générer des matrices aléatoires"""
        n = self.spin_n.value()
        m = self.spin_m.value()
        p = self.spin_p.value()
        
        matrices = {
            'A': np.random.randn(n, n) * 0.5,
            'B': np.random.randn(n, m),
            'C': np.random.randn(p, n),
            'D': np.random.randn(p, m) * 0.1
        }
        
        self.set_matrices(matrices)
        QMessageBox.information(self, "Génération", "Matrices aléatoires générées avec succès!")
    
    def load_examples(self):
        """Charger des exemples prédéfinis"""
        # Exemple : système masse-ressort-amortisseur
        examples = {
            'A': np.array([[0, 1], [-2, -3]]),
            'B': np.array([[0], [1]]),
            'C': np.array([[1, 0]]),
            'D': np.array([[0]])
        }
        
        self.spin_n.setValue(2)
        self.spin_m.setValue(1)
        self.spin_p.setValue(1)
        
        self.update_dimensions()
        self.set_matrices(examples)
        QMessageBox.information(self, "Exemple", "Système masse-ressort-amortisseur chargé!")
    
    def clear_matrices(self):
        """Effacer toutes les matrices"""
        for table in self.tables.values():
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    item = table.item(i, j)
                    if item:
                        item.setText("0.0000")
                        item.setBackground(Qt.white)
        
        self.emit_matrices_changed()
        QMessageBox.information(self, "Effacement", "Toutes les matrices ont été effacées.")
    
    def import_csv(self):
        """Importer des matrices depuis CSV"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Importer matrices", "", "Fichiers CSV (*.csv)"
        )
        if filename:
            # TODO: Implémenter l'import CSV
            QMessageBox.information(self, "Import", "Fonctionnalité d'import à implémenter")
    
    def validate_matrices(self):
        """Valider les matrices"""
        matrices = self.get_matrices()
        
        errors = []
        warnings = []
        
        # Vérifications de base
        A = matrices.get('A')
        if A is not None and A.shape[0] != A.shape[1]:
            errors.append("❌ La matrice A doit être carrée")
        
        B = matrices.get('B')
        C = matrices.get('C')
        D = matrices.get('D')
        
        if A is not None and B is not None and A.shape[0] != B.shape[0]:
            errors.append("❌ Incompatibilité dimensions A et B")
        
        if A is not None and C is not None and A.shape[1] != C.shape[1]:
            errors.append("❌ Incompatibilité dimensions A et C")
        
        if B is not None and D is not None and B.shape[1] != D.shape[1]:
            errors.append("❌ Incompatibilité dimensions B et D")
        
        if C is not None and D is not None and C.shape[0] != D.shape[0]:
            errors.append("❌ Incompatibilité dimensions C et D")
        
        # Vérifications de stabilité et autres propriétés
        if A is not None:
            eigenvals = np.linalg.eigvals(A)
            if np.any(np.real(eigenvals) > 0):
                warnings.append("⚠️ Le système est instable (pôles à partie réelle positive)")
            if np.any(np.abs(eigenvals) > 100):
                warnings.append("⚠️ Valeurs propres très grandes détectées")
        
        # Affichage des résultats
        if errors:
            QMessageBox.critical(self, "Erreurs de validation", "\n".join(errors))
        elif warnings:
            msg = "✅ Matrices valides !\n\nAvertissements:\n" + "\n".join(warnings)
            QMessageBox.warning(self, "Validation avec avertissements", msg)
        else:
            QMessageBox.information(self, "Validation", "✅ Toutes les matrices sont valides !")
            self.emit_matrices_changed()


class PlotCanvas(QWidget):
    """Canvas matplotlib intégré pour les graphiques avec thème amélioré"""
    
    def __init__(self, title: str = "Graphique"):
        super().__init__()
        self.title = title
        self.setup_ui()
        
    def setup_ui(self):
        """Configuration du canvas matplotlib"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Figure matplotlib avec thème
        self.figure = Figure(figsize=(10, 6), facecolor='#F0F8FF')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        
        layout.addWidget(self.canvas)
        
        # Boutons de contrôle avec style amélioré
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        btn_clear = QPushButton("🗑️ Effacer")
        btn_clear.setObjectName("controlButton")
        btn_clear.setMinimumHeight(35)
        btn_clear.clicked.connect(self.clear_plot)
        controls_layout.addWidget(btn_clear)
        
        btn_save = QPushButton("📷 Sauvegarder")
        btn_save.setObjectName("captureButton")
        btn_save.setMinimumHeight(35)
        btn_save.clicked.connect(self.save_figure)
        controls_layout.addWidget(btn_save)
        
        btn_grid = QPushButton("⊞ Grille")
        btn_grid.setObjectName("controlButton")
        btn_grid.setMinimumHeight(35)
        btn_grid.setCheckable(True)
        btn_grid.setChecked(True)
        btn_grid.clicked.connect(self.toggle_grid)
        controls_layout.addWidget(btn_grid)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
    
    def get_axes(self, subplot: Tuple[int, int, int] = (1, 1, 1)):
        """Obtenir les axes pour tracer"""
        ax = self.figure.add_subplot(*subplot)
        
        # Style par défaut amélioré
        ax.set_facecolor('#F0F8FF')
        ax.grid(True, alpha=0.3, color='#0D47A1')
        ax.spines['top'].set_color('#0D47A1')
        ax.spines['right'].set_color('#0D47A1')
        ax.spines['bottom'].set_color('#0D47A1')
        ax.spines['left'].set_color('#0D47A1')
        
        return ax
    
    def clear_plot(self):
        """Effacer le graphique"""
        self.figure.clear()
        self.canvas.draw()
    
    def toggle_grid(self):
        """Basculer l'affichage de la grille"""
        for ax in self.figure.get_axes():
            ax.grid(not ax.axes.xaxis._gridOnMajor)
        self.canvas.draw()
    
    def save_figure(self):
        """Sauvegarder la figure"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder le graphique", f"{self.title}.png",
            "Images PNG (*.png);;Images SVG (*.svg);;Images PDF (*.pdf)"
        )
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                QMessageBox.information(self, "Sauvegarde", f"Graphique sauvegardé: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def refresh(self):
        """Rafraîchir le canvas"""
        self.canvas.draw()


class ParameterPanel(QGroupBox):
    """Panneau de paramètres avec sliders et inputs amélioré"""
    
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, title: str = "Paramètres"):
        super().__init__(title)
        self.parameters = {}
        self.widgets = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Configuration de l'interface"""
        self.layout = QGridLayout(self)
        self.layout.setSpacing(15)
        self.layout.setColumnStretch(1, 1)
        
    def add_parameter(self, name: str, label: str, value: float, 
                     min_val: float = 0.0, max_val: float = 100.0, 
                     decimals: int = 2, suffix: str = ""):
        """Ajouter un paramètre avec style amélioré"""
        row = len(self.widgets)
        
        # Label avec style
        label_widget = QLabel(label)
        label_widget.setObjectName("paramLabel")
        self.layout.addWidget(label_widget, row, 0)
        
        # SpinBox avec style amélioré
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setDecimals(decimals)
        spinbox.setValue(value)
        spinbox.setMinimumHeight(30)
        spinbox.setMinimumWidth(120)
        if suffix:
            spinbox.setSuffix(f" {suffix}")
        spinbox.valueChanged.connect(lambda v, n=name: self.on_parameter_changed(n, v))
        
        self.layout.addWidget(spinbox, row, 1)
        self.widgets[name] = spinbox
        self.parameters[name] = value
    
    def on_parameter_changed(self, name: str, value: float):
        """Gestion du changement de paramètre"""
        self.parameters[name] = value
        self.parameters_changed.emit(self.parameters.copy())
    
    def get_parameters(self) -> Dict[str, float]:
        """Récupérer les paramètres"""
        return self.parameters.copy()
    
    def set_parameter(self, name: str, value: float):
        """Définir un paramètre"""
        if name in self.widgets:
            self.widgets[name].setValue(value)
            self.parameters[name] = value
