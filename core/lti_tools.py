#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils pour systèmes linéaires invariants dans le temps (LTI)
Calculs de pôles, zéros, réponses temporelles et fréquentielles
"""

import numpy as np
import scipy.linalg
import scipy.signal
from typing import Dict, Any, Optional, Tuple, Union
import control


def poles(A: np.ndarray) -> np.ndarray:
    """
    Calculer les pôles d'un système à partir de la matrice A
    
    Args:
        A: Matrice d'état (n×n)
        
    Returns:
        Pôles du système (valeurs propres de A)
    """
    return np.linalg.eigvals(A)


def zeros(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Calculer les zéros d'un système MIMO
    
    Args:
        A, B, C, D: Matrices du système
        
    Returns:
        Zéros du système
    """
    try:
        # Conversion en fonction de transfert pour calculer les zéros
        sys = control.ss(A, B, C, D)
        tf_sys = control.tf(sys)
        
        # Extraction des zéros
        if hasattr(tf_sys, 'zeros'):
            return tf_sys.zeros()
        else:
            # Pour les systèmes SISO
            if tf_sys.num[0][0] is not None:
                zeros_poly = np.roots(tf_sys.num[0][0])
                return zeros_poly
            else:
                return np.array([])
                
    except Exception as e:
        # Méthode alternative : résolution du problème de valeurs propres généralisé
        try:
            n, m = A.shape[0], B.shape[1]
            p = C.shape[0]
            
            if m == 1 and p == 1:  # Système SISO
                # Matrice système pour les zéros
                system_matrix = np.block([
                    [A, B],
                    [C, D]
                ])
                
                # Matrice de sélection
                E_matrix = np.block([
                    [np.eye(n), np.zeros((n, m))],
                    [np.zeros((p, n)), np.zeros((p, m))]
                ])
                
                # Valeurs propres généralisées
                eigenvals = scipy.linalg.eigvals(system_matrix, E_matrix)
                
                # Filtrer les valeurs infinies et les valeurs propres de A
                finite_eigs = eigenvals[np.isfinite(eigenvals)]
                system_poles = poles(A)
                
                # Zéros = valeurs propres qui ne sont pas des pôles
                zeros_list = []
                for eig in finite_eigs:
                    if not np.any(np.abs(eig - system_poles) < 1e-10):
                        zeros_list.append(eig)
                
                return np.array(zeros_list)
            else:
                # Pour les systèmes MIMO, retourner un tableau vide
                return np.array([])
                
        except:
            return np.array([])


def time_response(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                 u_type: str, t_final: float, dt: float = 0.01) -> Dict[str, Any]:
    """
    Calculer la réponse temporelle d'un système
    
    Args:
        A, B, C, D: Matrices du système
        u_type: Type d'entrée ('Échelon unitaire', 'Impulsion', 'Sinusoïde', 'Rampe')
        t_final: Temps final de simulation
        dt: Pas d'échantillonnage
        
    Returns:
        Dictionnaire contenant temps, entrée, sortie et états
    """
    # Vecteur temps
    t = np.arange(0, t_final + dt, dt)
    
    # Génération du signal d'entrée
    if u_type == "Échelon unitaire":
        u = np.ones((len(t), B.shape[1]))
    elif u_type == "Impulsion":
        u = np.zeros((len(t), B.shape[1]))
        if len(t) > 0:
            u[0, :] = 1.0 / dt  # Impulsion approximée
    elif u_type == "Sinusoïde":
        freq = 0.5  # 0.5 Hz
        u = np.sin(2 * np.pi * freq * t[:, np.newaxis])
        u = np.tile(u, (1, B.shape[1]))
    elif u_type == "Rampe":
        u = t[:, np.newaxis]
        u = np.tile(u, (1, B.shape[1]))
    else:
        # Entrée personnalisée (échelon par défaut)
        u = np.ones((len(t), B.shape[1]))
    
    # Simulation du système
    try:
        # Utilisation de scipy.signal.lsim
        if B.shape[1] == 1:  # SISO
            u_sim = u[:, 0]
        else:  # MIMO
            u_sim = u
        
        # Conversion en système control
        sys = control.ss(A, B, C, D)
        
        # Simulation
        t_out, y_out = control.forced_response(sys, t, u_sim.T)
        
        # Calcul des états (approximation)
        if A.shape[0] <= 10:  # Éviter les calculs trop coûteux
            x_states = simulate_states(A, B, u_sim, dt)
        else:
            x_states = np.zeros((len(t), A.shape[0]))
        
        return {
            'time': t_out,
            'input': u_sim,
            'output': y_out.T if y_out.ndim > 1 else y_out,
            'states': x_states,
            'success': True
        }
        
    except Exception as e:
        # Simulation alternative avec intégration numérique
        return simulate_system_numerical(A, B, C, D, u, t, dt)


def simulate_states(A: np.ndarray, B: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Simuler l'évolution des états
    
    Args:
        A, B: Matrices du système
        u: Signal d'entrée
        dt: Pas d'échantillonnage
        
    Returns:
        États du système
    """
    n = A.shape[0]
    N = len(u)
    x = np.zeros((N, n))
    
    # Matrices de transition discrètes
    Ad = scipy.linalg.expm(A * dt)
    Bd = np.linalg.inv(A) @ (Ad - np.eye(n)) @ B if np.linalg.det(A) != 0 else B * dt
    
    # Simulation itérative
    for k in range(1, N):
        if u.ndim == 1:
            u_k = u[k-1]
        else:
            u_k = u[k-1, :]
        
        x[k, :] = Ad @ x[k-1, :] + Bd @ u_k
    
    return x


def simulate_system_numerical(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                             u: np.ndarray, t: np.ndarray, dt: float) -> Dict[str, Any]:
    """
    Simulation numérique alternative du système
    
    Args:
        A, B, C, D: Matrices du système
        u: Signal d'entrée
        t: Vecteur temps
        dt: Pas d'échantillonnage
        
    Returns:
        Dictionnaire des résultats
    """
    try:
        from scipy.integrate import solve_ivp
        
        def system_dynamics(t_val, x, u_func):
            u_val = u_func(t_val)
            return A @ x + B @ u_val
        
        # Fonction d'interpolation pour l'entrée
        def u_interp(t_val):
            idx = min(int(t_val / dt), len(u) - 1)
            if u.ndim == 1:
                return u[idx]
            else:
                return u[idx, :]
        
        # Conditions initiales
        x0 = np.zeros(A.shape[0])
        
        # Résolution
        sol = solve_ivp(
            lambda t_val, x: system_dynamics(t_val, x, u_interp),
            [t[0], t[-1]],
            x0,
            t_eval=t,
            method='RK45'
        )
        
        if sol.success:
            x_states = sol.y.T
            y_output = np.array([C @ x + D @ u_interp(t_val) for t_val, x in zip(t, x_states)])
            
            return {
                'time': t,
                'input': u,
                'output': y_output,
                'states': x_states,
                'success': True
            }
        else:
            raise Exception("Échec de l'intégration numérique")
            
    except Exception as e:
        # Retour d'erreur
        return {
            'time': t,
            'input': u,
            'output': np.zeros((len(t), C.shape[0])),
            'states': np.zeros((len(t), A.shape[0])),
            'success': False,
            'error': str(e)
        }


def bode_data(sys, w: np.ndarray) -> Dict[str, Any]:
    """
    Calculer les données du diagramme de Bode
    
    Args:
        sys: Système (control.TransferFunction ou control.StateSpace)
        w: Vecteur de pulsations
        
    Returns:
        Dictionnaire contenant fréquence, module et phase
    """
    try:
        # Calcul de la réponse en fréquence
        mag, phase, omega = control.bode_plot(sys, w, plot=False)
        
        # Conversion en dB et degrés
        magnitude_db = 20 * np.log10(mag)
        phase_deg = phase * 180 / np.pi
        
        return {
            'frequency': omega,
            'magnitude': magnitude_db,
            'phase': phase_deg,
            'magnitude_linear': mag,
            'phase_rad': phase
        }
        
    except Exception as e:
        # Retour d'erreur avec données vides
        return {
            'frequency': w,
            'magnitude': np.zeros_like(w),
            'phase': np.zeros_like(w),
            'magnitude_linear': np.ones_like(w),
            'phase_rad': np.zeros_like(w),
            'error': str(e)
        }


def step_info(sys) -> Dict[str, float]:
    """
    Calculer les informations sur la réponse indicielle
    
    Args:
        sys: Système (control object)
        
    Returns:
        Dictionnaire avec les caractéristiques de la réponse
    """
    try:
        # Réponse indicielle
        t, y = control.step_response(sys)
        
        # Valeur finale
        steady_state = y[-1] if len(y) > 0 else 0
        
        # Temps de montée (10% à 90%)
        if steady_state != 0:
            idx_10 = np.where(y >= 0.1 * steady_state)[0]
            idx_90 = np.where(y >= 0.9 * steady_state)[0]
            
            rise_time = t[idx_90[0]] - t[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else 0.0
        else:
            rise_time = 0.0
        
        # Dépassement
        max_value = np.max(y) if len(y) > 0 else 0
        overshoot = max(0, (max_value - steady_state) / abs(steady_state) * 100) if steady_state != 0 else 0
        
        # Temps de pic
        peak_idx = np.argmax(y) if len(y) > 0 else 0
        peak_time = t[peak_idx] if len(t) > peak_idx else 0
        
        # Temps d'établissement (±2%)
        if steady_state != 0:
            tolerance = 0.02 * abs(steady_state)
            settling_indices = np.where(np.abs(y - steady_state) <= tolerance)[0]
            settling_time = t[settling_indices[0]] if len(settling_indices) > 0 else t[-1]
        else:
            settling_time = t[-1] if len(t) > 0 else 0
        
        return {
            'rise_time': rise_time,
            'overshoot': overshoot,
            'peak_time': peak_time,
            'settling_time': settling_time,
            'steady_state_value': steady_state,
            'peak_value': max_value
        }
        
    except Exception as e:
        return {
            'rise_time': 0.0,
            'overshoot': 0.0,
            'peak_time': 0.0,
            'settling_time': 0.0,
            'steady_state_value': 0.0,
            'peak_value': 0.0,
            'error': str(e)
        }


def nyquist_data(sys, w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculer les données du diagramme de Nyquist
    
    Args:
        sys: Système
        w: Vecteur de pulsations (optionnel)
        
    Returns:
        Dictionnaire avec les données de Nyquist
    """
    try:
        if w is None:
            w = np.logspace(-2, 2, 1000)
        
        # Réponse en fréquence
        mag, phase, omega = control.bode_plot(sys, w, plot=False)
        
        # Coordonnées polaires vers cartésiennes
        real_part = mag * np.cos(phase)
        imag_part = mag * np.sin(phase)
        
        return {
            'frequency': omega,
            'real': real_part,
            'imaginary': imag_part,
            'magnitude': mag,
            'phase': phase
        }
        
    except Exception as e:
        return {
            'frequency': w if w is not None else np.array([]),
            'real': np.array([]),
            'imaginary': np.array([]),
            'magnitude': np.array([]),
            'phase': np.array([]),
            'error': str(e)
        }


def stability_margins(sys) -> Dict[str, float]:
    """
    Calculer les marges de stabilité
    
    Args:
        sys: Système en boucle ouverte
        
    Returns:
        Marges de gain et de phase
    """
    try:
        gm, pm, wg, wp = control.margin(sys)
        
        # Conversion en dB pour la marge de gain
        gm_db = 20 * np.log10(gm) if gm > 0 else float('-inf')
        
        return {
            'gain_margin_db': gm_db,
            'gain_margin_linear': gm,
            'phase_margin_deg': pm,
            'gain_crossover_freq': wg,
            'phase_crossover_freq': wp
        }
        
    except Exception as e:
        return {
            'gain_margin_db': float('inf'),
            'gain_margin_linear': float('inf'),
            'phase_margin_deg': 90.0,
            'gain_crossover_freq': 0.0,
            'phase_crossover_freq': 0.0,
            'error': str(e)
        }


def root_locus_data(sys, k_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculer les données du lieu des racines
    
    Args:
        sys: Système
        k_values: Valeurs de gain (optionnel)
        
    Returns:
        Données du lieu des racines
    """
    try:
        if k_values is None:
            k_values = np.logspace(-2, 2, 1000)
        
        # Calcul du lieu des racines
        roots_array, k_out = control.root_locus(sys, k_values, plot=False)
        
        return {
            'gains': k_out,
            'roots': roots_array,
            'poles_open_loop': poles(sys.A) if hasattr(sys, 'A') else control.pole(sys),
            'zeros_open_loop': zeros(sys.A, sys.B, sys.C, sys.D) if hasattr(sys, 'A') else control.zero(sys)
        }
        
    except Exception as e:
        return {
            'gains': k_values if k_values is not None else np.array([]),
            'roots': np.array([]),
            'poles_open_loop': np.array([]),
            'zeros_open_loop': np.array([]),
            'error': str(e)
        }


def frequency_response(sys, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculer la réponse en fréquence
    
    Args:
        sys: Système
        w: Vecteur de pulsations
        
    Returns:
        Magnitude, phase, fréquences
    """
    try:
        mag, phase, omega = control.bode_plot(sys, w, plot=False)
        return mag, phase, omega
    except Exception as e:
        return np.ones_like(w), np.zeros_like(w), w


def impulse_response(sys, t: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculer la réponse impulsionnelle
    
    Args:
        sys: Système
        t: Vecteur temps (optionnel)
        
    Returns:
        Temps, réponse
    """
    try:
        if t is None:
            t_out, y_out = control.impulse_response(sys)
        else:
            t_out, y_out = control.impulse_response(sys, t)
        
        return t_out, y_out
        
    except Exception as e:
        if t is None:
            t = np.linspace(0, 10, 1000)
        return t, np.zeros_like(t)


def is_stable(sys) -> bool:
    """
    Vérifier la stabilité d'un système
    
    Args:
        sys: Système
        
    Returns:
        True si stable, False sinon
    """
    try:
        if hasattr(sys, 'A'):
            # Système d'état
            eigenvals = poles(sys.A)
        else:
            # Fonction de transfert
            eigenvals = control.pole(sys)
        
        return np.all(np.real(eigenvals) < 0)
        
    except:
        return False


def controllability_index(A: np.ndarray, B: np.ndarray) -> float:
    """
    Calculer un indice de contrôlabilité
    
    Args:
        A, B: Matrices du système
        
    Returns:
        Indice de contrôlabilité (0 à 1)
    """
    try:
        from core.ctrb_obsv import ctrb_matrix
        
        Wc = ctrb_matrix(A, B)
        rank_Wc = np.linalg.matrix_rank(Wc)
        n = A.shape[0]
        
        return rank_Wc / n
        
    except:
        return 0.0


def observability_index(A: np.ndarray, C: np.ndarray) -> float:
    """
    Calculer un indice d'observabilité
    
    Args:
        A, C: Matrices du système
        
    Returns:
        Indice d'observabilité (0 à 1)
    """
    try:
        from core.ctrb_obsv import obsv_matrix
        
        Wo = obsv_matrix(A, C)
        rank_Wo = np.linalg.matrix_rank(Wo)
        n = A.shape[0]
        
        return rank_Wo / n
        
    except:
        return 0.0
