#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conception de régulateurs PID
Méthodes de Ziegler-Nichols, Cohen-Coon, et simulation avec anti-windup
"""

import numpy as np
import control
from typing import Tuple, Dict, Any, Optional
from scipy.integrate import solve_ivp
from .lti_tools import step_info
import warnings

warnings.filterwarnings('ignore')


def pid_ziegler_nichols(system, method: str = 'step_response') -> Tuple[float, float, float]:
    """
    Calcul des gains PID selon la méthode de Ziegler-Nichols
    
    Args:
        system: Système de contrôle (control object)
        method: 'step_response' ou 'ultimate_gain'
        
    Returns:
        Tuple (Kp, Ki, Kd)
    """
    try:
        if method == 'step_response':
            return zn_step_response_method(system)
        elif method == 'ultimate_gain':
            return zn_ultimate_gain_method(system)
        else:
            return zn_step_response_method(system)
            
    except Exception as e:
        # Valeurs par défaut conservatrices
        return 1.0, 0.1, 0.01


def zn_step_response_method(system) -> Tuple[float, float, float]:
    """
    Méthode de Ziegler-Nichols basée sur la réponse indicielle
    
    Args:
        system: Système de contrôle
        
    Returns:
        Gains (Kp, Ki, Kd)
    """
    try:
        # Obtenir la réponse indicielle
        t, y = control.step_response(system, T=np.linspace(0, 50, 5000))
        
        if len(y) == 0 or np.all(y == 0):
            raise ValueError("Réponse indicielle nulle")
        
        # Calcul des paramètres caractéristiques
        params = calculate_step_response_parameters(system)
        
        # Extraction des paramètres nécessaires
        K = params.get('static_gain', 1.0)  # Gain statique
        L = estimate_dead_time(t, y)        # Temps mort (retard)
        T_tau = estimate_time_constant(t, y, K)  # Constante de temps
        
        if L <= 0:
            L = 0.1
        if T_tau <= 0:
            T_tau = 1.0
        if K == 0:
            K = 1.0
            
        # Calcul des gains selon Ziegler-Nichols (réponse indicielle)
        if abs(L/T_tau) < 0.1:
            # Système sans retard dominant - utiliser les règles modifiées
            Kp = 0.6 / K
            Ki = 2.0 / T_tau
            Kd = T_tau / 8.0
        else:
            # Système avec retard - règles classiques de ZN
            Kp = 1.2 * T_tau / (K * L)
            Ki = 0.5 / L
            Kd = 0.5 * L
        
        # Vérification des limites
        Kp = max(0.1, min(100.0, abs(Kp)))
        Ki = max(0.01, min(10.0, abs(Ki)))
        Kd = max(0.0, min(1.0, abs(Kd)))
        
        return Kp, Ki, Kd
        
    except Exception as e:
        # Paramètres par défaut basés sur l'analyse du système
        try:
            # Essayer d'extraire des informations basiques du système
            poles = control.pole(system)
            if len(poles) > 0:
                dominant_pole = min(abs(p.real) for p in poles if p.real < 0)
                Kp = 1.0 / dominant_pole if dominant_pole > 0 else 1.0
                Ki = dominant_pole / 10.0 if dominant_pole > 0 else 0.1
                Kd = 1.0 / (dominant_pole * 10) if dominant_pole > 0 else 0.01
                return abs(Kp), abs(Ki), abs(Kd)
        except:
            pass
        
        return 1.0, 0.1, 0.01


def zn_ultimate_gain_method(system) -> Tuple[float, float, float]:
    """
    Méthode de Ziegler-Nichols basée sur le gain ultime (méthode de Nyquist)
    
    Args:
        system: Système de contrôle
        
    Returns:
        Gains (Kp, Ki, Kd)
    """
    try:
        # Calcul du gain ultime et de la période ultime
        Ku, Tu = find_ultimate_gain_and_period(system)
        
        if Ku <= 0 or Tu <= 0:
            # Fallback vers la méthode de réponse indicielle
            return zn_step_response_method(system)
        
        # Calcul des gains selon les règles de Ziegler-Nichols
        Kp = 0.6 * Ku
        Ki = 2.0 * Kp / Tu
        Kd = Kp * Tu / 8.0
        
        # Vérification des limites
        Kp = max(0.1, min(100.0, Kp))
        Ki = max(0.01, min(10.0, Ki))
        Kd = max(0.0, min(1.0, Kd))
        
        return Kp, Ki, Kd
        
    except Exception as e:
        return zn_step_response_method(system)


def find_ultimate_gain_and_period(system) -> Tuple[float, float]:
    """
    Trouver le gain ultime et la période ultime
    
    Args:
        system: Système de contrôle
        
    Returns:
        (Ku, Tu) - Gain ultime et période ultime
    """
    try:
        # Calculer la marge de gain et la fréquence de croisement
        gm, pm, wg, wp = control.margin(system)
        
        if gm is not None and wg is not None and gm > 0 and wg > 0:
            Ku = gm
            Tu = 2 * np.pi / wg
            return Ku, Tu
        
        # Méthode alternative: recherche par balayage en fréquence
        w = np.logspace(-2, 2, 1000)
        mag, phase, omega = control.bode_plot(system, w, plot=False)
        
        # Trouver le croisement à -180°
        phase_deg = phase * 180 / np.pi
        idx_180 = np.where(np.abs(phase_deg + 180) < 5)[0]
        
        if len(idx_180) > 0:
            wg_est = omega[idx_180[0]]
            Ku_est = 1.0 / mag[idx_180[0]]  # Inverse du module au croisement
            Tu_est = 2 * np.pi / wg_est
            return Ku_est, Tu_est
        
        # Valeurs par défaut si pas de croisement trouvé
        return 2.0, 1.0
        
    except Exception as e:
        return 2.0, 1.0


def pid_cohen_coon(system) -> Tuple[float, float, float]:
    """
    Calcul des gains PID selon la méthode de Cohen-Coon
    
    Args:
        system: Système de contrôle
        
    Returns:
        Tuple (Kp, Ki, Kd)
    """
    try:
        # Obtenir la réponse indicielle
        t, y = control.step_response(system, T=np.linspace(0, 50, 5000))
        
        if len(y) == 0 or np.all(y == 0):
            raise ValueError("Réponse indicielle nulle")
        
        # Paramètres du système
        K = y[-1] if len(y) > 0 and abs(y[-1]) > 1e-10 else 1.0  # Gain statique
        L = estimate_dead_time(t, y)  # Temps mort
        T_tau = estimate_time_constant(t, y, K)  # Constante de temps
        
        if L <= 0:
            L = 0.1
        if T_tau <= 0:
            T_tau = 1.0
        if K == 0:
            K = 1.0
        
        # Ratio caractéristique
        alpha = L / T_tau
        
        # Formules de Cohen-Coon pour PID
        Kp = (1.35 / K) * (T_tau / L) * (1 + 0.18 * alpha)
        Ti = L * (2.5 - 2.0 * alpha) / (1 - 0.39 * alpha)  # Temps intégral
        Td = L * (0.37 - 0.37 * alpha) / (1 - 0.81 * alpha)  # Temps dérivé
        
        # Conversion en gains Ki et Kd
        Ki = Kp / Ti if Ti > 0 else 0
        Kd = Kp * Td
        
        # Vérification des limites et correction si nécessaire
        if alpha > 1.0:  # Système avec retard dominant
            # Utiliser des formules modifiées pour éviter l'instabilité
            Kp = 0.9 / K * (T_tau / L)
            Ki = 0.3 / L
            Kd = 0.3 * L
        
        # Limites finales
        Kp = max(0.1, min(50.0, abs(Kp)))
        Ki = max(0.01, min(5.0, abs(Ki)))
        Kd = max(0.0, min(2.0, abs(Kd)))
        
        return Kp, Ki, Kd
        
    except Exception as e:
        # Fallback vers Ziegler-Nichols
        return zn_step_response_method(system)


def calculate_step_response_parameters(system) -> Dict[str, float]:
    """
    Calculer les paramètres de la réponse indicielle
    
    Args:
        system: Système de contrôle
        
    Returns:
        Dictionnaire des paramètres
    """
    try:
        # Utiliser la fonction existante de lti_tools
        from core.lti_tools import step_info
        return step_info(system)
        
    except Exception as e:
        # Calcul manuel
        try:
            t, y = control.step_response(system, T=np.linspace(0, 20, 2000))
            
            if len(y) == 0:
                return {'static_gain': 1.0, 'rise_time': 1.0, 'settling_time': 5.0}
            
            # Gain statique
            static_gain = y[-1] if abs(y[-1]) > 1e-10 else 1.0
            
            # Temps de montée (10% à 90%)
            if static_gain != 0:
                idx_10 = np.where(y >= 0.1 * static_gain)[0]
                idx_90 = np.where(y >= 0.9 * static_gain)[0]
                rise_time = t[idx_90[0]] - t[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else 1.0
            else:
                rise_time = 1.0
            
            # Dépassement
            max_value = np.max(y) if len(y) > 0 else static_gain
            overshoot = max(0, (max_value - static_gain) / abs(static_gain) * 100) if static_gain != 0 else 0
            
            # Temps d'établissement (±2%)
            if static_gain != 0:
                settling_mask = np.abs(y - static_gain) <= 0.02 * abs(static_gain)
                settling_indices = np.where(settling_mask)[0]
                settling_time = t[settling_indices[0]] if len(settling_indices) > 0 else t[-1]
            else:
                settling_time = t[-1]
            
            return {
                'static_gain': static_gain,
                'rise_time': rise_time,
                'overshoot': overshoot,
                'settling_time': settling_time,
                'peak_time': t[np.argmax(y)] if len(y) > 0 else 1.0,
                'peak_value': max_value
            }
            
        except:
            return {'static_gain': 1.0, 'rise_time': 1.0, 'settling_time': 5.0}


def estimate_dead_time(t: np.ndarray, y: np.ndarray) -> float:
    """
    Estimer le temps mort d'un système
    
    Args:
        t: Vecteur temps
        y: Réponse du système
        
    Returns:
        Temps mort estimé
    """
    try:
        if len(y) == 0 or len(t) == 0:
            return 0.1
        
        # Méthode: temps où la sortie atteint 5% de sa valeur finale
        final_value = y[-1]
        
        if abs(final_value) < 1e-10:
            return 0.1
        
        threshold = 0.05 * abs(final_value)
        idx = np.where(np.abs(y) >= threshold)[0]
        
        if len(idx) > 0:
            dead_time = t[idx[0]]
        else:
            dead_time = 0.1
        
        return max(0.01, min(5.0, dead_time))
        
    except:
        return 0.1


def estimate_time_constant(t: np.ndarray, y: np.ndarray, K: float) -> float:
    """
    Estimer la constante de temps d'un système
    
    Args:
        t: Vecteur temps
        y: Réponse du système
        K: Gain statique
        
    Returns:
        Constante de temps estimée
    """
    try:
        if len(y) == 0 or abs(K) < 1e-10:
            return 1.0
        
        # Méthode: temps pour atteindre 63.2% de la valeur finale
        threshold = 0.632 * abs(K)
        idx = np.where(np.abs(y) >= threshold)[0]
        
        if len(idx) > 0:
            time_constant = t[idx[0]]
        else:
            # Méthode alternative: tangente à l'origine
            if len(y) > 10:
                # Calculer la dérivée au début de la réponse
                dy_dt = np.gradient(y, t)
                initial_slope = dy_dt[len(dy_dt)//10]  # Éviter les points initiaux bruités
                
                if abs(initial_slope) > 1e-10:
                    time_constant = abs(K) / abs(initial_slope)
                else:
                    time_constant = 1.0
            else:
                time_constant = 1.0
        
        return max(0.1, min(20.0, time_constant))
        
    except:
        return 1.0


def simulate_pid_control(system, Kp: float, Ki: float, Kd: float,
                        t_final: float = 20.0, dt: float = 0.01,
                        setpoint_amplitude: float = 1.0, setpoint_time: float = 1.0,
                        antiwindup: Dict[str, Any] = None,
                        disturbance: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Simuler un système avec régulateur PID
    
    Args:
        system: Système à contrôler
        Kp, Ki, Kd: Gains PID
        t_final: Temps final de simulation
        dt: Pas de temps
        setpoint_amplitude: Amplitude de la consigne
        setpoint_time: Temps d'application de la consigne
        antiwindup: Configuration anti-windup
        disturbance: Perturbations externes
        
    Returns:
        Dictionnaire des résultats de simulation
    """
    try:
        # Configuration par défaut de l'anti-windup
        if antiwindup is None:
            antiwindup = {'enabled': False, 'limit': 10.0}
        
        # Vecteur temps
        t = np.arange(0, t_final + dt, dt)
        N = len(t)
        
        # Signal de consigne
        setpoint = np.zeros(N)
        setpoint[t >= setpoint_time] = setpoint_amplitude
        
        # Perturbations
        if disturbance is None:
            disturbance = np.zeros(N)
        elif len(disturbance) != N:
            disturbance = np.zeros(N)
        
        # Conversion du système en espace d'état si nécessaire
        if hasattr(system, 'A'):
            A, B, C, D = system.A, system.B, system.C, system.D
        else:
            ss_sys = control.ss(system)
            A, B, C, D = ss_sys.A, ss_sys.B, ss_sys.C, ss_sys.D
        
        # Dimensions
        n_states = A.shape[0]
        
        # Initialisation des variables
        x = np.zeros(n_states)  # États du système
        output = np.zeros(N)
        control_signal = np.zeros(N)
        error_signal = np.zeros(N)
        integral_error = 0.0
        previous_error = 0.0
        
        # Simulation
        for k in range(N):
            # Sortie du système
            y = C @ x + D * (control_signal[k-1] if k > 0 else 0) + disturbance[k]
            if np.isscalar(y):
                output[k] = y
            else:
                output[k] = y[0]  # Prendre la première sortie pour MIMO
            
            # Erreur
            error = setpoint[k] - output[k]
            error_signal[k] = error
            
            # Terme intégral
            integral_error += error * dt
            
            # Anti-windup sur le terme intégral
            if antiwindup['enabled']:
                integral_limit = antiwindup['limit'] / (abs(Ki) + 1e-10)
                integral_error = max(-integral_limit, min(integral_limit, integral_error))
            
            # Terme dérivé
            derivative_error = (error - previous_error) / dt if k > 0 else 0.0
            
            # Signal de commande PID
            u = Kp * error + Ki * integral_error + Kd * derivative_error
            
            # Anti-windup sur le signal de commande
            if antiwindup['enabled']:
                u_limited = max(-antiwindup['limit'], min(antiwindup['limit'], u))
                
                # Back-calculation anti-windup
                if u != u_limited and abs(Ki) > 1e-10:
                    integral_error -= (u - u_limited) / Ki * dt
                
                u = u_limited
            
            control_signal[k] = u
            
            # Évolution des états du système (méthode d'Euler)
            if k < N - 1:
                u_input = u if B.shape[1] == 1 else np.array([u])
                x_dot = A @ x + B.flatten() * u if B.shape[1] == 1 else A @ x + B @ u_input
                x = x + x_dot * dt
            
            # Mise à jour pour la dérivée
            previous_error = error
        
        # Calcul des performances
        performance = calculate_pid_performance(t, setpoint, output, error_signal, control_signal)
        
        return {
            'success': True,
            'time': t,
            'setpoint': setpoint,
            'output': output,
            'control': control_signal,
            'error': error_signal,
            'performance': performance
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time': np.arange(0, t_final + dt, dt),
            'setpoint': np.ones(int(t_final/dt) + 1),
            'output': np.zeros(int(t_final/dt) + 1),
            'control': np.zeros(int(t_final/dt) + 1),
            'error': np.ones(int(t_final/dt) + 1)
        }


def calculate_pid_performance(t: np.ndarray, setpoint: np.ndarray, 
                             output: np.ndarray, error: np.ndarray,
                             control: np.ndarray) -> Dict[str, float]:
    """
    Calculer les performances du contrôleur PID
    
    Args:
        t: Vecteur temps
        setpoint: Signal de consigne
        output: Signal de sortie
        error: Signal d'erreur
        control: Signal de commande
        
    Returns:
        Dictionnaire des performances
    """
    try:
        # Valeur finale de la consigne
        final_setpoint = setpoint[-1] if len(setpoint) > 0 else 1.0
        
        # Temps de montée (10% à 90% de la valeur finale)
        if abs(final_setpoint) > 1e-10:
            idx_10 = np.where(output >= 0.1 * final_setpoint)[0]
            idx_90 = np.where(output >= 0.9 * final_setpoint)[0]
            rise_time = t[idx_90[0]] - t[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else 0
        else:
            rise_time = 0
        
        # Dépassement
        max_output = np.max(output) if len(output) > 0 else final_setpoint
        overshoot = max(0, (max_output - final_setpoint) / abs(final_setpoint) * 100) if final_setpoint != 0 else 0
        
        # Temps d'établissement (±2%)
        if abs(final_setpoint) > 1e-10:
            tolerance = 0.02 * abs(final_setpoint)
            settling_mask = np.abs(output - final_setpoint) <= tolerance
            settling_indices = np.where(settling_mask)[0]
            settling_time = t[settling_indices[0]] if len(settling_indices) > 0 else t[-1]
        else:
            settling_time = t[-1] if len(t) > 0 else 0
        
        # Erreur statique
        steady_state_error = abs(np.mean(error[-100:])) if len(error) > 100 else abs(error[-1]) if len(error) > 0 else 0
        
        # Critères intégraux
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        # ISE (Integral of Squared Error)
        ise = np.trapz(error**2, dx=dt) if len(error) > 0 else 0
        
        # IAE (Integral of Absolute Error)
        iae = np.trapz(np.abs(error), dx=dt) if len(error) > 0 else 0
        
        # ITAE (Integral of Time-weighted Absolute Error)
        itae = np.trapz(t * np.abs(error), dx=dt) if len(error) > 0 else 0
        
        # Effort de commande
        control_effort = np.trapz(np.abs(control), dx=dt) if len(control) > 0 else 0
        max_control = np.max(np.abs(control)) if len(control) > 0 else 0
        
        return {
            'rise_time': rise_time,
            'overshoot_percent': overshoot,
            'settling_time': settling_time,
            'steady_state_error': steady_state_error,
            'peak_time': t[np.argmax(output)] if len(output) > 0 else 0,
            'ise': ise,
            'iae': iae,
            'itae': itae,
            'control_effort': control_effort,
            'max_control': max_control
        }
        
    except Exception as e:
        return {
            'rise_time': 0,
            'overshoot_percent': 0,
            'settling_time': 0,
            'steady_state_error': float('inf'),
            'peak_time': 0,
            'ise': float('inf'),
            'iae': float('inf'),
            'itae': float('inf'),
            'control_effort': 0,
            'max_control': 0
        }


def pid_manual_tuning(system, specifications: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Réglage manuel optimisé basé sur des spécifications
    
    Args:
        system: Système à contrôler
        specifications: Dict avec 'overshoot_max', 'settling_time_max', 'rise_time_max'
        
    Returns:
        Gains (Kp, Ki, Kd) optimisés
    """
    try:
        # Gains initiaux par Ziegler-Nichols
        Kp_init, Ki_init, Kd_init = pid_ziegler_nichols(system)
        
        # Spécifications par défaut
        overshoot_max = specifications.get('overshoot_max', 10.0)
        settling_time_max = specifications.get('settling_time_max', 5.0)
        rise_time_max = specifications.get('rise_time_max', 2.0)
        
        # Optimisation simple par ajustement des gains
        best_gains = (Kp_init, Ki_init, Kd_init)
        best_score = float('inf')
        
        # Grille de recherche autour des gains initiaux
        kp_range = np.linspace(0.5 * Kp_init, 2.0 * Kp_init, 5)
        ki_range = np.linspace(0.1 * Ki_init, 2.0 * Ki_init, 5)
        kd_range = np.linspace(0.0, 2.0 * Kd_init, 5)
        
        for Kp in kp_range:
            for Ki in ki_range:
                for Kd in kd_range:
                    # Simulation rapide
                    try:
                        result = simulate_pid_control(system, Kp, Ki, Kd, t_final=10.0)
                        
                        if result['success']:
                            perf = result['performance']
                            
                            # Fonction de coût
                            cost = 0
                            
                            # Pénalité dépassement
                            if perf['overshoot_percent'] > overshoot_max:
                                cost += (perf['overshoot_percent'] - overshoot_max) ** 2
                            
                            # Pénalité temps d'établissement
                            if perf['settling_time'] > settling_time_max:
                                cost += (perf['settling_time'] - settling_time_max) ** 2
                            
                            # Pénalité temps de montée
                            if perf['rise_time'] > rise_time_max:
                                cost += (perf['rise_time'] - rise_time_max) ** 2
                            
                            # Pénalité erreur statique
                            cost += perf['steady_state_error'] * 10
                            
                            # Pénalité effort de commande excessif
                            cost += perf['max_control'] * 0.01
                            
                            if cost < best_score:
                                best_score = cost
                                best_gains = (Kp, Ki, Kd)
                    
                    except:
                        continue
        
        return best_gains
        
    except Exception as e:
        # Fallback vers Ziegler-Nichols
        return pid_ziegler_nichols(system)


def tune_pid_genetic_algorithm(system, specifications: Dict[str, float],
                              population_size: int = 20, generations: int = 50) -> Tuple[float, float, float]:
    """
    Réglage PID par algorithme génétique (version simplifiée)
    
    Args:
        system: Système à contrôler
        specifications: Spécifications de performance
        population_size: Taille de la population
        generations: Nombre de générations
        
    Returns:
        Gains optimaux (Kp, Ki, Kd)
    """
    try:
        # Initialisation de la population
        population = []
        for _ in range(population_size):
            Kp = np.random.uniform(0.1, 10.0)
            Ki = np.random.uniform(0.01, 5.0)
            Kd = np.random.uniform(0.0, 2.0)
            population.append([Kp, Ki, Kd])
        
        # Évolution
        for generation in range(generations):
            # Évaluation de la fitness
            fitness_scores = []
            
            for individual in population:
                Kp, Ki, Kd = individual
                
                try:
                    result = simulate_pid_control(system, Kp, Ki, Kd, t_final=15.0)
                    
                    if result['success']:
                        perf = result['performance']
                        
                        # Calcul de la fitness (plus bas = meilleur)
                        fitness = 0
                        fitness += perf['overshoot_percent'] * 0.1
                        fitness += perf['settling_time']
                        fitness += perf['rise_time'] * 0.5
                        fitness += perf['steady_state_error'] * 10
                        fitness += perf['ise'] * 0.01
                        
                        fitness_scores.append(fitness)
                    else:
                        fitness_scores.append(1000.0)  # Très mauvaise fitness
                        
                except:
                    fitness_scores.append(1000.0)
            
            # Sélection des meilleurs individus
            sorted_indices = np.argsort(fitness_scores)
            elite_size = population_size // 4
            elite = [population[i] for i in sorted_indices[:elite_size]]
            
            # Nouvelle population
            new_population = elite.copy()
            
            # Reproduction et mutation
            while len(new_population) < population_size:
                # Sélection de deux parents
                parent1 = elite[np.random.randint(0, len(elite))]
                parent2 = elite[np.random.randint(0, len(elite))]
                
                # Croisement
                alpha = np.random.random()
                child = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
                
                # Mutation
                mutation_rate = 0.1
                if np.random.random() < mutation_rate:
                    gene_idx = np.random.randint(0, 3)
                    child[gene_idx] += np.random.normal(0, child[gene_idx] * 0.1)
                    child[gene_idx] = max(0.01, child[gene_idx])  # Éviter les valeurs négatives
                
                new_population.append(child)
            
            population = new_population
        
        # Retourner le meilleur individu
        final_fitness = []
        for individual in population:
            Kp, Ki, Kd = individual
            try:
                result = simulate_pid_control(system, Kp, Ki, Kd, t_final=10.0)
                if result['success']:
                    perf = result['performance']
                    fitness = perf['ise'] + perf['overshoot_percent'] * 0.1 + perf['settling_time']
                    final_fitness.append(fitness)
                else:
                    final_fitness.append(1000.0)
            except:
                final_fitness.append(1000.0)
        
        best_idx = np.argmin(final_fitness)
        return tuple(population[best_idx])
        
    except Exception as e:
        # Fallback vers la méthode manuelle
        return pid_manual_tuning(system, specifications)
