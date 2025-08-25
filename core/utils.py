#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires et validations pour ControlSysLab
Exceptions personnalisées, validations de systèmes et fonctions utilitaires
"""

import numpy as np
import re
import ast
from typing import Tuple, Dict, Any, List, Optional, Callable, Union
import warnings


class ControlSysLabError(Exception):
    """Exception de base pour ControlSysLab"""
    pass


class SystemValidationError(ControlSysLabError):
    """Erreur de validation de système"""
    pass


class MatrixDimensionError(ControlSysLabError):
    """Erreur de dimension de matrice"""
    pass


class NonlinearSystemError(ControlSysLabError):
    """Erreur de système non-linéaire"""
    pass


class PIDDesignError(ControlSysLabError):
    """Erreur de conception PID"""
    pass


def validate_system_matrices(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                            tolerance: float = 1e-10) -> Tuple[bool, List[str]]:
    """
    Valider les matrices d'un système linéaire
    
    Args:
        A, B, C, D: Matrices du système
        tolerance: Tolérance numérique
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Vérifier que ce sont des arrays numpy
        if not all(isinstance(mat, np.ndarray) for mat in [A, B, C, D]):
            errors.append("Toutes les matrices doivent être des arrays NumPy")
            return False, errors
        
        # Vérifier les dimensions minimales
        if A.ndim != 2 or B.ndim != 2 or C.ndim != 2 or D.ndim != 2:
            errors.append("Toutes les matrices doivent être bidimensionnelles")
            return False, errors
        
        # Dimensions
        n, n_check = A.shape
        n_B, m = B.shape
        p, n_C = C.shape
        p_D, m_D = D.shape
        
        # Vérifications de compatibilité des dimensions
        if n != n_check:
            errors.append(f"La matrice A doit être carrée (actuellement {n}×{n_check})")
        
        if n_B != n:
            errors.append(f"La matrice B doit avoir {n} lignes (actuellement {n_B})")
        
        if n_C != n:
            errors.append(f"La matrice C doit avoir {n} colonnes (actuellement {n_C})")
        
        if p_D != p:
            errors.append(f"La matrice D doit avoir {p} lignes (actuellement {p_D})")
        
        if m_D != m:
            errors.append(f"La matrice D doit avoir {m} colonnes (actuellement {m_D})")
        
        # Vérifier les valeurs numériques
        for name, mat in [("A", A), ("B", B), ("C", C), ("D", D)]:
            if not np.isfinite(mat).all():
                errors.append(f"La matrice {name} contient des valeurs non finies (NaN ou Inf)")
            
            if np.any(np.abs(mat) > 1e12):
                errors.append(f"La matrice {name} contient des valeurs très grandes (> 1e12)")
        
        # Vérifications spécifiques au système
        if len(errors) == 0:
            # Vérifier la stabilité numérique
            try:
                eigenvals = np.linalg.eigvals(A)
                if np.any(np.abs(eigenvals) > 1e6):
                    errors.append("⚠️ Avertissement: Le système a des pôles très éloignés de l'origine")
            except:
                errors.append("⚠️ Impossible de calculer les valeurs propres de A")
            
            # Vérifier le conditionnement
            try:
                cond_A = np.linalg.cond(A)
                if cond_A > 1e12:
                    errors.append("⚠️ Avertissement: La matrice A est mal conditionnée")
            except:
                pass
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Erreur lors de la validation: {str(e)}")
        return False, errors


def validate_nonlinear_system(equations_text: str) -> Tuple[bool, Optional[Callable], str]:
    """
    Valider et compiler les équations d'un système non-linéaire
    
    Args:
        equations_text: Code Python décrivant le système
        
    Returns:
        (is_valid, compiled_function, error_message)
    """
    try:
        # Nettoyer le texte d'entrée
        clean_text = equations_text.strip()
        
        if not clean_text:
            return False, None, "Le code d'équations est vide"
        
        # Enlever les commentaires et lignes vides
        lines = []
        for line in clean_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        if not lines:
            return False, None, "Aucune équation valide trouvée"
        
        # Vérifications de sécurité basiques
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', '__import__',
            'eval(', 'exec(', 'open(', 'file(', 'input(', 'raw_input(',
            'globals(', 'locals(', 'vars(', 'dir(', 'getattr', 'setattr',
            'delattr', 'hasattr'
        ]
        
        code_lower = clean_text.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, None, f"Code non sécurisé détecté: {pattern}"
        
        # Construire la fonction du système
        function_template = """
def nonlinear_system_function(t, x, u=None):
    import numpy as np
    import math
    from math import sin, cos, tan, exp, log, sqrt, pi, e
    
    # Variables d'état
    if len(x) >= 1:
        x1 = x[0]
    if len(x) >= 2:
        x2 = x[1] 
    if len(x) >= 3:
        x3 = x[2]
    if len(x) >= 4:
        x4 = x[3]
    if len(x) >= 5:
        x5 = x[4]
    
    # Paramètres par défaut
    mu = 1.0
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    alpha = 1.0
    gamma = 1.0
    delta = 1.0
    
    # Variables d'entrée
    if u is not None and len(u) > 0:
        u1 = u[0]
    else:
        u1 = 0.0
    
    # Équations utilisateur (remplacées dynamiquement)
    {user_equations}
    
    # Retourner les dérivées
    if len(x) == 1:
        return np.array([dx1_dt])
    elif len(x) == 2:
        return np.array([dx1_dt, dx2_dt])
    elif len(x) == 3:
        return np.array([dx1_dt, dx2_dt, dx3_dt])
    elif len(x) == 4:
        return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])
    elif len(x) == 5:
        return np.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt])
    else:
        # Système générique
        result = []
        for i in range(len(x)):
            try:
                result.append(locals()[f'dx{i+1}_dt'])
            except KeyError:
                result.append(0.0)
        return np.array(result)
"""
        
        # Parser et transformer les équations utilisateur
        user_eqs = []
        for line in lines:
            if '=' in line and 'dt' in line:
                # Ligne d'équation différentielle
                user_eqs.append("    " + line)
            elif line.strip():
                # Autre ligne (paramètre, calcul intermédiaire)
                user_eqs.append("    " + line)
        
        if not user_eqs:
            # Équations par défaut pour test
            user_eqs = [
                "    dx1_dt = x2",
                "    dx2_dt = -x1 - 0.1*x2"
            ]
        
        # Construire le code final
        final_code = function_template.format(user_equations='\n'.join(user_eqs))
        
        # Compiler et tester
        namespace = {}
        exec(final_code, namespace)
        
        compiled_function = namespace['nonlinear_system_function']
        
        # Test de base
        test_x = np.array([1.0, 0.0])
        test_result = compiled_function(0.0, test_x)
        
        if not isinstance(test_result, np.ndarray):
            return False, None, "La fonction doit retourner un array NumPy"
        
        if len(test_result) != len(test_x):
            return False, None, f"Dimension incorrecte: attendu {len(test_x)}, obtenu {len(test_result)}"
        
        if not np.isfinite(test_result).all():
            return False, None, "La fonction produit des valeurs non finies"
        
        return True, compiled_function, "Validation réussie"
        
    except SyntaxError as e:
        return False, None, f"Erreur de syntaxe Python: {str(e)}"
    except Exception as e:
        return False, None, f"Erreur lors de la validation: {str(e)}"


def format_number(value: Union[int, float, complex], precision: int = 4) -> str:
    """
    Formater un nombre pour l'affichage
    
    Args:
        value: Nombre à formater
        precision: Nombre de décimales
        
    Returns:
        Chaîne formatée
    """
    if isinstance(value, complex):
        if abs(value.imag) < 1e-10:
            return format_number(value.real, precision)
        else:
            sign = '+' if value.imag >= 0 else '-'
            return f"{value.real:.{precision}f} {sign} {abs(value.imag):.{precision}f}j"
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        if abs(value) < 1e-10:
            return "0"
        elif abs(value) >= 1e6 or abs(value) <= 1e-4:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    else:
        return str(value)


def format_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    """
    Formater une matrice pour l'affichage
    
    Args:
        matrix: Matrice à formater
        precision: Précision d'affichage
        
    Returns:
        Chaîne formatée
    """
    if matrix.ndim == 1:
        # Vecteur
        elements = [format_number(val, precision) for val in matrix]
        return "[" + "  ".join(elements) + "]"
    elif matrix.ndim == 2:
        # Matrice
        lines = []
        for row in matrix:
            elements = [format_number(val, precision) for val in row]
            lines.append("[" + "  ".join(f"{elem:>10}" for elem in elements) + "]")
        return "\n".join(lines)
    else:
        return str(matrix)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sécurisée avec gestion de la division par zéro
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si division par zéro
        
    Returns:
        Résultat de la division ou valeur par défaut
    """
    if abs(denominator) < 1e-15:
        return default
    return numerator / denominator


def ensure_array_2d(arr: np.ndarray) -> np.ndarray:
    """
    S'assurer qu'un array est 2D
    
    Args:
        arr: Array d'entrée
        
    Returns:
        Array 2D
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 0:
        return arr.reshape(1, 1)
    else:
        return arr


def validate_pid_parameters(Kp: float, Ki: float, Kd: float) -> Tuple[bool, List[str]]:
    """
    Valider les paramètres PID
    
    Args:
        Kp, Ki, Kd: Gains PID
        
    Returns:
        (is_valid, warnings)
    """
    warnings_list = []
    is_valid = True
    
    # Vérifications de base
    if not all(np.isfinite([Kp, Ki, Kd])):
        is_valid = False
        warnings_list.append("❌ Les gains PID doivent être des nombres finis")
        return is_valid, warnings_list
    
    # Vérifications des plages recommandées
    if Kp < 0:
        warnings_list.append("⚠️ Gain proportionnel négatif peut causer l'instabilité")
    elif Kp > 100:
        warnings_list.append("⚠️ Gain proportionnel très élevé (>100)")
    elif Kp < 0.01:
        warnings_list.append("⚠️ Gain proportionnel très faible (<0.01)")
    
    if Ki < 0:
        warnings_list.append("⚠️ Gain intégral négatif peut causer l'instabilité")
    elif Ki > 10:
        warnings_list.append("⚠️ Gain intégral très élevé (>10)")
    
    if Kd < 0:
        warnings_list.append("⚠️ Gain dérivé négatif inhabituel")
    elif Kd > 5:
        warnings_list.append("⚠️ Gain dérivé très élevé (>5) - risque de bruit")
    
    # Vérifications de cohérence
    if Ki > 0 and Kp / Ki < 0.1:
        warnings_list.append("⚠️ Rapport Kp/Ki très faible - risque d'oscillations")
    
    if Kd > 0 and Kd / Kp > 1:
        warnings_list.append("⚠️ Rapport Kd/Kp > 1 - risque d'amplification du bruit")
    
    return is_valid, warnings_list


def check_system_stability(A: np.ndarray, tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Vérifier la stabilité d'un système linéaire
    
    Args:
        A: Matrice d'état
        tolerance: Tolérance numérique
        
    Returns:
        Dictionnaire avec les informations de stabilité
    """
    try:
        eigenvals = np.linalg.eigvals(A)
        
        # Stabilité asymptotique
        real_parts = np.real(eigenvals)
        is_stable = np.all(real_parts < -tolerance)
        
        # Stabilité marginale
        is_marginally_stable = np.all(real_parts <= tolerance) and not is_stable
        
        # Classification des pôles
        stable_poles = np.sum(real_parts < -tolerance)
        unstable_poles = np.sum(real_parts > tolerance)
        marginal_poles = len(eigenvals) - stable_poles - unstable_poles
        
        return {
            'is_stable': is_stable,
            'is_marginally_stable': is_marginally_stable,
            'eigenvalues': eigenvals,
            'stable_poles_count': stable_poles,
            'unstable_poles_count': unstable_poles,
            'marginal_poles_count': marginal_poles,
            'stability_margin': -np.max(real_parts) if len(real_parts) > 0 else 0.0
        }
        
    except Exception as e:
        return {
            'is_stable': False,
            'error': str(e)
        }


def estimate_system_bandwidth(A: np.ndarray, C: np.ndarray) -> float:
    """
    Estimer la bande passante d'un système
    
    Args:
        A: Matrice d'état
        C: Matrice de sortie
        
    Returns:
        Bande passante estimée (rad/s)
    """
    try:
        eigenvals = np.linalg.eigvals(A)
        
        # Pôles dominants (plus lents)
        stable_poles = eigenvals[np.real(eigenvals) < 0]
        
        if len(stable_poles) == 0:
            return 1.0  # Valeur par défaut
        
        # Pôle dominant = pôle le plus proche de l'axe imaginaire
        dominant_pole = stable_poles[np.argmax(np.real(stable_poles))]
        
        # Bande passante approximative
        if np.imag(dominant_pole) != 0:
            # Pôle complexe - utiliser la partie réelle
            bandwidth = abs(np.real(dominant_pole))
        else:
            # Pôle réel
            bandwidth = abs(dominant_pole)
        
        return bandwidth
        
    except Exception:
        return 1.0


def validate_control_input(u: np.ndarray, constraints: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
    """
    Valider un signal de commande
    
    Args:
        u: Signal de commande
        constraints: Contraintes (min_value, max_value, max_rate)
        
    Returns:
        (is_valid, violations)
    """
    violations = []
    
    if not isinstance(u, np.ndarray):
        violations.append("Le signal de commande doit être un array NumPy")
        return False, violations
    
    if not np.isfinite(u).all():
        violations.append("Signal de commande contient des valeurs non finies")
        return False, violations
    
    if constraints:
        # Contraintes d'amplitude
        if 'min_value' in constraints:
            if np.any(u < constraints['min_value']):
                violations.append(f"Signal sous la limite minimale ({constraints['min_value']})")
        
        if 'max_value' in constraints:
            if np.any(u > constraints['max_value']):
                violations.append(f"Signal au-dessus de la limite maximale ({constraints['max_value']})")
        
        # Contraintes de vitesse
        if 'max_rate' in constraints and len(u) > 1:
            du_dt = np.diff(u)
            max_rate_actual = np.max(np.abs(du_dt))
            if max_rate_actual > constraints['max_rate']:
                violations.append(f"Vitesse de variation trop élevée ({max_rate_actual:.2f} > {constraints['max_rate']})")
    
    return len(violations) == 0, violations


def create_time_vector(t_final: float, dt: float, t_start: float = 0.0) -> np.ndarray:
    """
    Créer un vecteur temps avec gestion des erreurs
    
    Args:
        t_final: Temps final
        dt: Pas de temps
        t_start: Temps initial
        
    Returns:
        Vecteur temps
    """
    if t_final <= t_start:
        raise ValueError("Le temps final doit être supérieur au temps initial")
    
    if dt <= 0:
        raise ValueError("Le pas de temps doit être positif")
    
    if dt > (t_final - t_start):
        raise ValueError("Le pas de temps est trop grand")
    
    # Calculer le nombre de points pour éviter les erreurs d'arrondi
    n_points = int(np.ceil((t_final - t_start) / dt)) + 1
    
    return np.linspace(t_start, t_final, n_points)


def interpolate_data(t_old: np.ndarray, data_old: np.ndarray, 
                    t_new: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpoler des données temporelles
    
    Args:
        t_old: Ancien vecteur temps
        data_old: Anciennes données
        t_new: Nouveau vecteur temps
        method: Méthode d'interpolation
        
    Returns:
        Données interpolées
    """
    try:
        from scipy.interpolate import interp1d
        
        if method == 'linear':
            f = interp1d(t_old, data_old, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            f = interp1d(t_old, data_old, kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
        else:
            f = interp1d(t_old, data_old, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
        
        return f(t_new)
        
    except ImportError:
        # Interpolation linéaire simple si scipy n'est pas disponible
        return np.interp(t_new, t_old, data_old)
    except Exception:
        # Fallback: répéter la dernière valeur
        return np.full_like(t_new, data_old[-1] if len(data_old) > 0 else 0.0)


def detect_outliers(data: np.ndarray, method: str = 'iqr', factor: float = 1.5) -> np.ndarray:
    """
    Détecter les valeurs aberrantes
    
    Args:
        data: Données à analyser
        method: Méthode de détection ('iqr', 'zscore')
        factor: Facteur de seuil
        
    Returns:
        Masque booléen des valeurs aberrantes
    """
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > factor
    
    else:
        return np.zeros(len(data), dtype=bool)


def compute_performance_metrics(time: np.ndarray, reference: np.ndarray, 
                               output: np.ndarray) -> Dict[str, float]:
    """
    Calculer les métriques de performance d'un système
    
    Args:
        time: Vecteur temps
        reference: Signal de référence
        output: Signal de sortie
        
    Returns:
        Dictionnaire des métriques
    """
    try:
        error = reference - output
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        
        # Erreurs intégrales
        iae = np.trapz(np.abs(error), dx=dt)  # Integral Absolute Error
        ise = np.trapz(error**2, dx=dt)       # Integral Squared Error
        itae = np.trapz(time * np.abs(error), dx=dt)  # Integral Time Absolute Error
        
        # Erreur statique
        steady_state_error = np.mean(np.abs(error[-min(100, len(error)):])) if len(error) > 0 else 0
        
        # Dépassement
        final_value = reference[-1] if len(reference) > 0 else 1.0
        max_output = np.max(output) if len(output) > 0 else 0
        overshoot = max(0, (max_output - final_value) / abs(final_value) * 100) if final_value != 0 else 0
        
        # Temps de réponse (approximatif)
        if final_value != 0:
            settling_mask = np.abs(output - final_value) <= 0.02 * abs(final_value)
            settling_indices = np.where(settling_mask)[0]
            settling_time = time[settling_indices[0]] if len(settling_indices) > 0 else time[-1]
        else:
            settling_time = time[-1] if len(time) > 0 else 0
        
        return {
            'iae': iae,
            'ise': ise,
            'itae': itae,
            'steady_state_error': steady_state_error,
            'overshoot_percent': overshoot,
            'settling_time': settling_time,
            'rms_error': np.sqrt(ise / len(time)) if len(time) > 0 else 0,
            'max_error': np.max(np.abs(error)) if len(error) > 0 else 0
        }
        
    except Exception as e:
        return {
            'iae': float('inf'),
            'ise': float('inf'),
            'itae': float('inf'),
            'steady_state_error': float('inf'),
            'overshoot_percent': 0,
            'settling_time': 0,
            'rms_error': float('inf'),
            'max_error': float('inf'),
            'error': str(e)
        }


def sanitize_filename(filename: str) -> str:
    """
    Nettoyer un nom de fichier
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        Nom de fichier nettoyé
    """
    # Caractères interdits dans les noms de fichiers
    invalid_chars = r'[<>:"/\\|?*]'
    
    # Remplacer les caractères interdits
    clean_name = re.sub(invalid_chars, '_', filename)
    
    # Limiter la longueur
    if len(clean_name) > 200:
        clean_name = clean_name[:200]
    
    # S'assurer qu'il n'est pas vide
    if not clean_name.strip():
        clean_name = "controlsyslab_file"
    
    return clean_name.strip()


def log_system_info(system_data: Dict[str, Any], operation: str) -> None:
    """
    Logger les informations système (version simplifiée)
    
    Args:
        system_data: Données système
        operation: Opération effectuée
    """
    try:
        # Log basique vers la console
        print(f"[ControlSysLab] {operation}")
        
        if 'A' in system_data and system_data['A'] is not None:
            n = system_data['A'].shape[0]
            print(f"  - Système d'ordre {n}")
        
        if 'poles' in system_data:
            poles = system_data['poles']
            if isinstance(poles, np.ndarray) and len(poles) > 0:
                stable = np.all(np.real(poles) < 0)
                print(f"  - Stabilité: {'Stable' if stable else 'Instable'}")
        
    except Exception:
        pass  # Logger silencieux en cas d'erreur


# Constantes utiles
MATLAB_TO_PYTHON_FUNCTIONS = {
    'abs': 'np.abs',
    'sqrt': 'np.sqrt',
    'exp': 'np.exp',
    'log': 'np.log',
    'log10': 'np.log10',
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'pi': 'np.pi',
    'inf': 'np.inf',
    'nan': 'np.nan'
}

DEFAULT_TOLERANCES = {
    'eigenvalue': 1e-10,
    'rank': 1e-12,
    'lyapunov': 1e-8,
    'pid_validation': 1e-6,
    'simulation': 1e-9
}

SYSTEM_LIMITS = {
    'max_dimension': 20,
    'max_simulation_time': 1000.0,
    'max_eigenvalue_magnitude': 1e6,
    'max_matrix_element': 1e10,
    'min_time_step': 1e-6,
    'max_time_step': 10.0
}
