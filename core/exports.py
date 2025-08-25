#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions d'exportation
Export CSV, PDF, sauvegarde de figures et génération de rapports
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import base64


def export_csv(data: Dict[str, Any], filename: str, 
              include_metadata: bool = True) -> bool:
    """
    Exporter des données vers un fichier CSV
    
    Args:
        data: Dictionnaire contenant les données à exporter
        filename: Nom du fichier de sortie
        include_metadata: Inclure les métadonnées
        
    Returns:
        True si l'export a réussi
    """
    try:
        # Créer un DataFrame pandas
        df_data = {}
        
        # Traiter les différents types de données
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    df_data[key] = value
                elif value.ndim == 2 and value.shape[1] == 1:
                    df_data[key] = value.flatten()
                elif value.ndim == 2:
                    # Matrices multiples colonnes
                    for i in range(value.shape[1]):
                        df_data[f"{key}_col_{i+1}"] = value[:, i]
                else:
                    # Array multidimensionnel - aplatir
                    df_data[key] = value.flatten()
            elif isinstance(value, (list, tuple)):
                df_data[key] = np.array(value)
            elif isinstance(value, (int, float, complex)):
                # Valeurs scalaires - répéter pour correspondre à la longueur
                max_length = max([len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 
                                for v in data.values() if isinstance(v, (np.ndarray, list, tuple))] + [1])
                df_data[key] = [value] * max_length
            elif isinstance(value, str):
                # Chaînes - traiter comme métadonnées
                continue
        
        # Créer le DataFrame
        if df_data:
            # S'assurer que toutes les colonnes ont la même longueur
            lengths = [len(v) for v in df_data.values()]
            if lengths and max(lengths) > 0:
                max_len = max(lengths)
                for key in df_data:
                    if len(df_data[key]) < max_len:
                        # Pad avec NaN
                        padded = np.full(max_len, np.nan)
                        padded[:len(df_data[key])] = df_data[key]
                        df_data[key] = padded
            
            df = pd.DataFrame(df_data)
        else:
            # DataFrame vide avec métadonnées seulement
            df = pd.DataFrame()
        
        # Ajouter métadonnées comme commentaires
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if include_metadata:
                f.write(f"# Exporté le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Source: ControlSysLab\n")
                
                # Métadonnées du système
                for key, value in data.items():
                    if isinstance(value, str):
                        f.write(f"# {key}: {value}\n")
                    elif isinstance(value, (int, float, complex)) and key not in df_data:
                        f.write(f"# {key}: {value}\n")
                
                f.write("#\n")
            
            # Écrire le DataFrame
            if not df.empty:
                df.to_csv(f, index=False, float_format='%.6f')
            else:
                f.write("# Aucune donnée numérique à exporter\n")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'export CSV: {str(e)}")
        return False


def export_current_figure(filename: str, dpi: int = 300, 
                         format: str = 'png', bbox_inches: str = 'tight') -> bool:
    """
    Exporter la figure matplotlib active
    
    Args:
        filename: Nom du fichier
        dpi: Résolution
        format: Format de l'image
        bbox_inches: Mode de recadrage
        
    Returns:
        True si l'export a réussi
    """
    try:
        # Obtenir la figure active
        fig = plt.gcf()
        
        if fig is None:
            return False
        
        # Déterminer le format à partir de l'extension si non spécifié
        if '.' in filename:
            ext = filename.split('.')[-1].lower()
            if ext in ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps']:
                format = ext
        
        # Sauvegarder
        fig.savefig(filename, dpi=dpi, format=format, 
                   bbox_inches=bbox_inches, facecolor='white', 
                   edgecolor='none', transparent=False)
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de l'export de figure: {str(e)}")
        return False


def save_figure(figure: Figure, filename: str, dpi: int = 300) -> bool:
    """
    Sauvegarder une figure matplotlib spécifique
    
    Args:
        figure: Figure matplotlib
        filename: Nom du fichier
        dpi: Résolution
        
    Returns:
        True si la sauvegarde a réussi
    """
    try:
        figure.savefig(filename, dpi=dpi, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        return True
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {str(e)}")
        return False


def export_pdf_report(system_data: Dict[str, Any], 
                     figure_paths: List[str],
                     output_path: str,
                     include_analysis: bool = True) -> bool:
    """
    Générer un rapport PDF complet
    
    Args:
        system_data: Données du système analysé
        figure_paths: Chemins vers les figures à inclure
        output_path: Chemin de sortie du PDF
        include_analysis: Inclure l'analyse détaillée
        
    Returns:
        True si la génération a réussi
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        # Créer le document
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0D47A1')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#0D47A1')
        )
        
        # Contenu du document
        story = []
        
        # Page de titre
        story.append(Paragraph("ControlSysLab", title_style))
        story.append(Paragraph("Rapport d'Analyse de Système de Contrôle", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Informations générales
        current_time = datetime.now().strftime('%d/%m/%Y à %H:%M:%S')
        story.append(Paragraph(f"<b>Généré le:</b> {current_time}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Informations système
        if system_data:
            story.append(Paragraph("Informations du Système", heading_style))
            
            # Créer un tableau pour les données système
            table_data = [['Paramètre', 'Valeur']]
            
            for key, value in system_data.items():
                if isinstance(value, str):
                    table_data.append([key, value])
                elif isinstance(value, (int, float)):
                    table_data.append([key, f"{value:.6f}"])
                elif isinstance(value, np.ndarray):
                    if value.size <= 16:  # Seulement pour les petites matrices
                        if value.ndim == 1:
                            value_str = str(value.tolist())
                        else:
                            value_str = f"Matrice {value.shape}"
                        table_data.append([key, value_str])
                elif isinstance(value, complex):
                    table_data.append([key, f"{value.real:.4f} + {value.imag:.4f}j"])
            
            if len(table_data) > 1:
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
        
        # Matrices du système
        if include_analysis and system_data:
            matrices = ['A', 'B', 'C', 'D']
            for matrix_name in matrices:
                if matrix_name in system_data:
                    matrix = system_data[matrix_name]
                    if isinstance(matrix, np.ndarray) and matrix.size > 0:
                        story.append(Paragraph(f"Matrice {matrix_name}", heading_style))
                        story.append(Paragraph(format_matrix_for_pdf(matrix), styles['Code']))
                        story.append(Spacer(1, 15))
        
        # Analyses et résultats
        if 'poles' in system_data:
            story.append(Paragraph("Analyse de Stabilité", heading_style))
            poles = system_data['poles']
            if isinstance(poles, np.ndarray):
                poles_str = format_poles_for_pdf(poles)
                story.append(Paragraph(f"<b>Pôles du système:</b><br/>{poles_str}", styles['Normal']))
                
                # Stabilité
                is_stable = np.all(np.real(poles) < 0)
                stability_text = "STABLE" if is_stable else "INSTABLE"
                color = "green" if is_stable else "red"
                story.append(Paragraph(f"<b>Stabilité:</b> <font color='{color}'>{stability_text}</font>", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Figures
        if figure_paths:
            story.append(PageBreak())
            story.append(Paragraph("Graphiques et Analyses", heading_style))
            
            for i, fig_path in enumerate(figure_paths):
                if os.path.exists(fig_path):
                    try:
                        # Ajouter l'image avec une taille appropriée
                        img = Image(fig_path)
                        img.drawHeight = 4*inch
                        img.drawWidth = 6*inch
                        story.append(img)
                        story.append(Paragraph(f"Figure {i+1}: Analyse du système", styles['Caption']))
                        story.append(Spacer(1, 20))
                    except Exception as e:
                        story.append(Paragraph(f"Erreur lors du chargement de la figure {fig_path}: {str(e)}", 
                                             styles['Normal']))
        
        # Pied de page
        story.append(Spacer(1, 30))
        story.append(Paragraph("─" * 80, styles['Normal']))
        story.append(Paragraph("Généré par ControlSysLab - Analyse et Conception de Systèmes de Contrôle", 
                             styles['Italic']))
        
        # Construire le PDF
        doc.build(story)
        
        return True
        
    except ImportError:
        print("ReportLab n'est pas installé. Génération d'un rapport texte simple.")
        return export_simple_text_report(system_data, figure_paths, output_path.replace('.pdf', '.txt'))
        
    except Exception as e:
        print(f"Erreur lors de la génération du rapport PDF: {str(e)}")
        return False


def export_simple_text_report(system_data: Dict[str, Any], 
                             figure_paths: List[str],
                             output_path: str) -> bool:
    """
    Générer un rapport texte simple (fallback si ReportLab n'est pas disponible)
    
    Args:
        system_data: Données du système
        figure_paths: Chemins des figures
        output_path: Chemin de sortie
        
    Returns:
        True si la génération a réussi
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CONTROLSYSLAB - RAPPORT D'ANALYSE\n")
            f.write("=" * 60 + "\n\n")
            
            current_time = datetime.now().strftime('%d/%m/%Y à %H:%M:%S')
            f.write(f"Généré le: {current_time}\n\n")
            
            if system_data:
                f.write("INFORMATIONS DU SYSTÈME\n")
                f.write("-" * 25 + "\n")
                
                for key, value in system_data.items():
                    if isinstance(value, str):
                        f.write(f"{key}: {value}\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.6f}\n")
                    elif isinstance(value, np.ndarray):
                        if value.ndim == 1 and len(value) <= 10:
                            f.write(f"{key}: {value}\n")
                        elif value.ndim == 2 and value.size <= 16:
                            f.write(f"{key}:\n{value}\n")
                        else:
                            f.write(f"{key}: Matrice {value.shape}\n")
                    elif isinstance(value, complex):
                        f.write(f"{key}: {value.real:.4f} + {value.imag:.4f}j\n")
                
                f.write("\n")
            
            # Analyses
            if 'poles' in system_data:
                f.write("ANALYSE DE STABILITÉ\n")
                f.write("-" * 20 + "\n")
                
                poles = system_data['poles']
                if isinstance(poles, np.ndarray):
                    f.write("Pôles du système:\n")
                    for i, pole in enumerate(poles):
                        if np.isreal(pole):
                            f.write(f"  p{i+1} = {pole.real:.6f}\n")
                        else:
                            sign = '+' if pole.imag >= 0 else '-'
                            f.write(f"  p{i+1} = {pole.real:.6f} {sign} {abs(pole.imag):.6f}j\n")
                    
                    is_stable = np.all(np.real(poles) < 0)
                    f.write(f"\nStabilité: {'STABLE' if is_stable else 'INSTABLE'}\n")
                
                f.write("\n")
            
            # Liste des figures
            if figure_paths:
                f.write("FIGURES GÉNÉRÉES\n")
                f.write("-" * 16 + "\n")
                for i, fig_path in enumerate(figure_paths):
                    if os.path.exists(fig_path):
                        f.write(f"Figure {i+1}: {os.path.basename(fig_path)}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("Fin du rapport\n")
            
        return True
        
    except Exception as e:
        print(f"Erreur lors de la génération du rapport texte: {str(e)}")
        return False


def format_matrix_for_pdf(matrix: np.ndarray) -> str:
    """
    Formater une matrice pour l'affichage PDF
    
    Args:
        matrix: Matrice numpy
        
    Returns:
        Chaîne formatée
    """
    if matrix.ndim == 1:
        return f"[{' '.join([f'{val:8.4f}' for val in matrix])}]"
    else:
        lines = []
        for row in matrix:
            line = "[" + "  ".join([f"{val:8.4f}" for val in row]) + "]"
            lines.append(line)
        return "<br/>".join(lines)


def format_poles_for_pdf(poles: np.ndarray) -> str:
    """
    Formater les pôles pour l'affichage PDF
    
    Args:
        poles: Array des pôles
        
    Returns:
        Chaîne formatée
    """
    formatted = []
    for i, pole in enumerate(poles):
        if np.isreal(pole):
            formatted.append(f"p{i+1} = {pole.real:.4f}")
        else:
            sign = '+' if pole.imag >= 0 else '-'
            formatted.append(f"p{i+1} = {pole.real:.4f} {sign} {abs(pole.imag):.4f}j")
    
    return "<br/>".join(formatted)


def export_simulation_data(simulation_results: Dict[str, Any], 
                         base_filename: str) -> List[str]:
    """
    Exporter les résultats de simulation vers plusieurs formats
    
    Args:
        simulation_results: Résultats de la simulation
        base_filename: Nom de base pour les fichiers
        
    Returns:
        Liste des fichiers créés
    """
    created_files = []
    
    try:
        # Export CSV des données temporelles
        if 'time' in simulation_results:
            csv_data = {
                'time': simulation_results['time']
            }
            
            # Ajouter toutes les données temporelles
            for key in ['output', 'input', 'states', 'control', 'error', 'setpoint']:
                if key in simulation_results:
                    data = simulation_results[key]
                    if isinstance(data, np.ndarray):
                        if data.ndim == 1:
                            csv_data[key] = data
                        elif data.ndim == 2:
                            for i in range(data.shape[1]):
                                csv_data[f"{key}_{i+1}"] = data[:, i]
            
            csv_filename = base_filename + "_data.csv"
            if export_csv(csv_data, csv_filename):
                created_files.append(csv_filename)
        
        # Export des paramètres et performances
        params_data = {}
        for key, value in simulation_results.items():
            if key not in ['time', 'output', 'input', 'states', 'control', 'error', 'setpoint']:
                if isinstance(value, (int, float, complex, str)):
                    params_data[key] = value
                elif isinstance(value, dict):
                    # Aplatir les dictionnaires imbriqués
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, complex, str)):
                            params_data[f"{key}_{subkey}"] = subvalue
        
        if params_data:
            params_filename = base_filename + "_parameters.csv"
            if export_csv(params_data, params_filename):
                created_files.append(params_filename)
        
        return created_files
        
    except Exception as e:
        print(f"Erreur lors de l'export des données de simulation: {str(e)}")
        return created_files


def create_comparison_report(systems_data: List[Dict[str, Any]], 
                           system_names: List[str],
                           output_path: str) -> bool:
    """
    Créer un rapport de comparaison entre plusieurs systèmes
    
    Args:
        systems_data: Liste des données de systèmes
        system_names: Noms des systèmes
        output_path: Chemin de sortie
        
    Returns:
        True si le rapport a été créé avec succès
    """
    try:
        comparison_data = {
            'system_name': system_names,
            'report_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Comparer les propriétés communes
        common_props = ['stability', 'settling_time', 'overshoot', 'rise_time']
        
        for prop in common_props:
            values = []
            for system_data in systems_data:
                if prop in system_data:
                    values.append(system_data[prop])
                else:
                    values.append(None)
            comparison_data[prop] = values
        
        # Export en CSV
        return export_csv(comparison_data, output_path, include_metadata=True)
        
    except Exception as e:
        print(f"Erreur lors de la création du rapport de comparaison: {str(e)}")
        return False


def backup_session_data(session_data: Dict[str, Any], 
                       backup_dir: str = "backups") -> Optional[str]:
    """
    Sauvegarder les données de session
    
    Args:
        session_data: Données de la session
        backup_dir: Répertoire de sauvegarde
        
    Returns:
        Chemin du fichier de sauvegarde créé
    """
    try:
        import json
        
        # Créer le répertoire de sauvegarde
        os.makedirs(backup_dir, exist_ok=True)
        
        # Nom du fichier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"controlsyslab_session_{timestamp}.json"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Convertir les arrays numpy en listes pour la sérialisation JSON
        serializable_data = convert_for_json(session_data)
        
        # Sauvegarder
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        return backup_path
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {str(e)}")
        return None


def convert_for_json(obj):
    """
    Convertir un objet pour la sérialisation JSON
    
    Args:
        obj: Objet à convertir
        
    Returns:
        Objet sérialisable JSON
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj
