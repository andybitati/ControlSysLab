#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ControlSysLab - Application principale
Point d'entrée de l'application desktop pour l'analyse et conception de systèmes de contrôle
"""

import sys
import os
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont

from .ui.main_window import MainWindow


class ControlSysLabApp:
    """Application principale ControlSysLab"""
    
    def __init__(self):
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        
    def setup_application(self) -> QApplication:
        """Configuration de l'application Qt"""
        app = QApplication(sys.argv)
        app.setApplicationName("ControlSysLab")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("ControlSysLab")
        
        # Configuration de la police par défaut
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        app.setFont(font)
        
        # Application du thème
        self.apply_theme(app)
        
        return app
    
    def apply_theme(self, app: QApplication) -> None:
        """Application du thème QSS"""
        try:
            qss_path = os.path.join(os.path.dirname(__file__), "themes", "app.qss")
            if os.path.exists(qss_path):
                with open(qss_path, 'r', encoding='utf-8') as f:
                    app.setStyleSheet(f.read())
            else:
                print(f"Fichier de thème non trouvé: {qss_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du thème: {e}")
    
    def run(self) -> int:
        """Lancement de l'application"""
        try:
            self.app = self.setup_application()
            self.main_window = MainWindow()
            self.main_window.show()
            
            return self.app.exec_()
            
        except Exception as e:
            QMessageBox.critical(
                None,
                "Erreur critique",
                f"Impossible de démarrer l'application:\n{str(e)}"
            )
            return 1


def main():
    """Point d'entrée principal"""
    app = ControlSysLabApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
