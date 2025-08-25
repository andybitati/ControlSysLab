#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface principale de ControlSysLab
Fenêtre principale avec sidebar, header et zone d'onglets
"""

import os
from typing import Dict, Any
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTabWidget, QLabel, QToolBar, QAction, QFileDialog, QMessageBox,
    QSplitter, QFrame, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap

from .tabs.tab_state_analysis import TabStateAnalysis
from .tabs.tab_ctrb_obsv import TabCtrbObsv
from .tabs.tab_state_feedback import TabStateFeedback
from .tabs.tab_output_feedback import TabOutputFeedback
from .tabs.tab_nonlinear import TabNonlinear
from .tabs.tab_pid import TabPID
from ..core.exports import export_current_figure, export_pdf_report


class MainWindow(QMainWindow):
    """Fenêtre principale de l'application"""
    
    # Signaux
    system_changed = pyqtSignal(dict)  # Émis quand le système change
    
    def __init__(self):
        super().__init__()
        self.current_system = {}
        self.tabs = {}
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        self.setWindowTitle("ControlSysLab - Analyse et Conception de Systèmes de Contrôle")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Création du splitter principal
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Sidebar
        sidebar = self.create_sidebar()
        splitter.addWidget(sidebar)
        
        # Zone principale
        main_area = self.create_main_area()
        splitter.addWidget(main_area)
        
        # Proportions du splitter
        splitter.setSizes([250, 1150])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        # Toolbar
        self.create_toolbar()
        
    def create_sidebar(self) -> QWidget:
        """Création de la barre latérale de navigation"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(250)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(15)
        
        # Titre
        title_label = QLabel("ControlSysLab")
        title_label.setObjectName("sidebarTitle")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Boutons de navigation
        nav_buttons = [
            ("Analyse d'État", "state_analysis"),
            ("Contrôlabilité/Observabilité", "ctrb_obsv"),
            ("Commande d'État", "state_feedback"),
            ("Commande de Sortie", "output_feedback"),
            ("Systèmes Non-Linéaires", "nonlinear"),
            ("Régulateurs PID", "pid")
        ]
        
        self.nav_buttons = {}
        for text, key in nav_buttons:
            btn = QPushButton(text)
            btn.setObjectName("navButton")
            btn.setMinimumHeight(45)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self.switch_tab(k))
            self.nav_buttons[key] = btn
            layout.addWidget(btn)
        
        # Espacement
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Bouton thème (inactif pour le moment)
        theme_btn = QPushButton("Thème Clair")
        theme_btn.setObjectName("themeButton")
        theme_btn.setEnabled(False)
        layout.addWidget(theme_btn)
        
        return sidebar
    
    def create_main_area(self) -> QWidget:
        """Création de la zone principale avec onglets"""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # En-tête
        header = self.create_header()
        layout.addWidget(header)
        
        # Zone des onglets
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabs")
        self.tab_widget.setTabsClosable(False)
        
        # Création des onglets
        self.tabs["state_analysis"] = TabStateAnalysis()
        self.tabs["ctrb_obsv"] = TabCtrbObsv()
        self.tabs["state_feedback"] = TabStateFeedback()
        self.tabs["output_feedback"] = TabOutputFeedback()
        self.tabs["nonlinear"] = TabNonlinear()
        self.tabs["pid"] = TabPID()
        
        # Ajout des onglets
        self.tab_widget.addTab(self.tabs["state_analysis"], "Analyse d'État")
        self.tab_widget.addTab(self.tabs["ctrb_obsv"], "Contrôlabilité/Observabilité")
        self.tab_widget.addTab(self.tabs["state_feedback"], "Commande d'État")
        self.tab_widget.addTab(self.tabs["output_feedback"], "Commande de Sortie")
        self.tab_widget.addTab(self.tabs["nonlinear"], "Systèmes Non-Linéaires")
        self.tab_widget.addTab(self.tabs["pid"], "Régulateurs PID")
        
        # Masquer les onglets (navigation par sidebar)
        tab_bar = self.tab_widget.tabBar()
        if tab_bar:
            tab_bar.setVisible(False)
        
        layout.addWidget(self.tab_widget)
        
        return main_widget
    
    def create_header(self) -> QWidget:
        """Création de l'en-tête bleu"""
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Titre de l'onglet actuel
        self.header_title = QLabel("Analyse d'État")
        self.header_title.setObjectName("headerTitle")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        self.header_title.setFont(header_font)
        layout.addWidget(self.header_title)
        
        # Espacement
        layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Boutons d'action rapide
        actions = [
            ("Nouveau", "nouveau"),
            ("Ouvrir", "ouvrir"),
            ("Enregistrer", "enregistrer"),
            ("Exporter", "exporter"),
            ("📷 Capturer", "capturer")
        ]
        
        for text, action in actions:
            btn = QPushButton(text)
            btn.setObjectName("headerActionButton" if action != "capturer" else "captureButton")
            btn.setMinimumHeight(35)
            btn.clicked.connect(lambda checked, a=action: self.handle_action(a))
            layout.addWidget(btn)
        
        return header
    
    def create_toolbar(self):
        """Création de la barre d'outils (optionnelle)"""
        pass
    
    def setup_connections(self):
        """Configuration des connexions de signaux"""
        # Connexion du signal de changement de système
        self.system_changed.connect(self.on_system_changed)
        
        # Connexions des onglets
        for tab in self.tabs.values():
            if hasattr(tab, 'system_changed'):
                tab.system_changed.connect(self.on_system_changed)
    
    def switch_tab(self, tab_key: str):
        """Basculer vers un onglet spécifique"""
        if tab_key not in self.tabs:
            return
            
        # Mise à jour des boutons de navigation
        for key, btn in self.nav_buttons.items():
            btn.setChecked(key == tab_key)
        
        # Basculer vers l'onglet
        tab_index = list(self.tabs.keys()).index(tab_key)
        self.tab_widget.setCurrentIndex(tab_index)
        
        # Mise à jour du titre
        titles = {
            "state_analysis": "Analyse d'État",
            "ctrb_obsv": "Contrôlabilité/Observabilité", 
            "state_feedback": "Commande d'État",
            "output_feedback": "Commande de Sortie",
            "nonlinear": "Systèmes Non-Linéaires",
            "pid": "Régulateurs PID"
        }
        self.header_title.setText(titles.get(tab_key, "ControlSysLab"))
    
    def handle_action(self, action: str):
        """Gestion des actions de l'en-tête"""
        if action == "nouveau":
            self.nouveau_systeme()
        elif action == "ouvrir":
            self.ouvrir_systeme()
        elif action == "enregistrer":
            self.enregistrer_systeme()
        elif action == "exporter":
            self.exporter_donnees()
        elif action == "capturer":
            self.capturer_graphique()
    
    def nouveau_systeme(self):
        """Créer un nouveau système"""
        self.current_system = {}
        self.system_changed.emit(self.current_system)
        QMessageBox.information(self, "Nouveau", "Nouveau système créé")
    
    def ouvrir_systeme(self):
        """Ouvrir un système existant"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un système", "", "Fichiers JSON (*.json)"
        )
        if filename:
            # TODO: Implémenter le chargement
            QMessageBox.information(self, "Ouvrir", f"Ouverture de {filename}")
    
    def enregistrer_systeme(self):
        """Enregistrer le système actuel"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le système", "", "Fichiers JSON (*.json)"
        )
        if filename:
            # TODO: Implémenter la sauvegarde
            QMessageBox.information(self, "Enregistrer", f"Sauvegarde vers {filename}")
    
    def exporter_donnees(self):
        """Exporter les données"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_data'):
                current_tab.export_data()
            else:
                export_pdf_report({}, [], "")
                QMessageBox.information(self, "Export", "Données exportées avec succès")
        except Exception as e:
            QMessageBox.warning(self, "Erreur d'export", f"Erreur lors de l'export: {str(e)}")
    
    def capturer_graphique(self):
        """Capturer le graphique actuel"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'capture_figure'):
                current_tab.capture_figure()
            else:
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Capturer le graphique", "", "Images PNG (*.png);;Images SVG (*.svg)"
                )
                if filename:
                    export_current_figure(filename)
                    QMessageBox.information(self, "Capture", f"Graphique sauvegardé: {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Erreur de capture", f"Erreur lors de la capture: {str(e)}")
    
    def on_system_changed(self, system_data: Dict[str, Any]):
        """Gestion du changement de système"""
        self.current_system = system_data
        # Propager aux autres onglets
        for tab in self.tabs.values():
            if hasattr(tab, 'update_system'):
                tab.update_system(system_data)
    
    def closeEvent(self, a0):
        """Gestion de la fermeture de l'application"""
        event = a0
        reply = QMessageBox.question(
            self, "Fermeture", "Voulez-vous vraiment fermer ControlSysLab?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
