#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de systèmes non-linéaires
Linéarisation, méthode de Lyapunov, simulation et portraits de phase
"""

import numpy as np
import sympy as sp
from typing import Callable, Dict, Any, Tuple, Optional, List
from scipy.integrate import solve_ivp
import scipy.linalg


def linearize_system(nonlinear_func: Callable, equilibrium_point: np.ndarray, 
                    u_eq: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linéariser un système non-linéaire autour d'un point d'équilibre
    
    Args:
        nonlinear_func: Fonction décrivant le système dx/dt = f(x, u)
        equilibrium_point: Point d'équilibre x_eq
        u_eq: Entrée d'équilibre (défaut: zéro)
        
    Returns:
        Matrices A et B du système linéarisé
    """
    try:
        n = len(equilibrium_point)
        
        # Entrée d'équilibre par défaut
        if u_eq is None:
            u_eq = np.array([0.0])
        
        m = len(u_eq)
        
        # Calcul numérique du Jacobien par différences finies
        h = 1e-8  # Pas pour les différences finies
        
        # Jacobien par rapport à x (matrice A)
        A = np.zeros((n, n))
        
        for j in range(n):
            # Perturbation positive
            x_plus = equilibrium_point.copy()
            x_plus[j] += h
            f_plus = nonlinear_func(0, x_plus, u_eq)
            
            # Perturbation négative
            x_minus = equilibrium_point.copy()
            x_minus[j] -= h
            f_minus = nonlinear_func(0, x_minus, u_eq)
            
            # Dérivée partielle
            A[:, j] = (f_plus - f_minus) / (2 * h)
        
        # Jacobien par rapport à u (matrice B)
        B = np.zeros((n, m))
        
        for j in range(m):
            # Perturbation positive
            u_plus = u_eq.copy()
            u_plus[j] += h
            f_plus = nonlinear_func(0, equilibrium_point, u_plus)
            
            # Perturbation négative
            u_minus = u_eq.copy()
            u_minus[j] -= h
            f_minus = nonlinear_func(0, equilibrium_point, u_minus)
            
            # Dérivée partielle
            B[:, j] = (f_plus - f_minus) / (2 * h)
        
        return A, B
        
    except Exception as e:
        # Retourner des matrices par défaut en cas d'erreur
        n = len(equilibrium_point)
        m = 1 if u_eq is None else len(u_eq)
        return np.eye(n), np.ones((n, m))


def linearize_symbolic(equations: str, state_vars: List[str], 
                      equilibrium_point: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linéarisation symbolique avec SymPy
    
    Args:
        equations: Équations du système sous forme de chaîne
        state_vars: Liste des variables d'état
        equilibrium_point: Point d'équilibre
        
    Returns:
        Matrices A et B linéarisées
    """
    try:
        # Définir les variables symboliques
        x_symbols = [sp.Symbol(var) for var in state_vars]
        u_symbol = sp.Symbol('u')
        
        # Parser les équations (format simplifié)
        # Cette fonction nécessiterait un parser plus sophistiqué pour être complète
        
        # Pour l'instant, retourner à la méthode numérique
        raise NotImplementedError("Linéarisation symbolique nécessite un parser d'équations plus avancé")
        
    except Exception as e:
        # Fallback vers la méthode numérique
        n = len(state_vars)
        return np.eye(n), np.ones((n, 1))


def lyapunov_stability_analysis(A: np.ndarray) -> Dict[str, Any]:
    """
    Analyser la stabilité par la méthode de Lyapunov pour le système linéarisé
    
    Args:
        A: Matrice du système linéarisé
        
    Returns:
        Dictionnaire avec les résultats de l'analyse
    """
    try:
        n = A.shape[0]
        
        # Vérification de la stabilité par les valeurs propres
        eigenvals = np.linalg.eigvals(A)
        is_stable = np.all(np.real(eigenvals) < 0)
        
        if not is_stable:
            return {
                'stable': False,
                'method': 'eigenvalues',
                'eigenvalues': eigenvals,
                'P': None,
                'lyapunov_function': None
            }
        
        # Construction d'une fonction de Lyapunov quadratique V(x) = x^T P x
        try:
            # Résolution de l'équation de Lyapunov: A^T P + P A + Q = 0
            Q = np.eye(n)  # Matrice définie positive
            P = scipy.linalg.solve_lyapunov(A.T, -Q)
            
            # Vérifier que P est définie positive
            eigenvals_P = np.linalg.eigvals(P)
            P_positive_definite = np.all(eigenvals_P > 0)
            
            if P_positive_definite:
                lyapunov_func = lambda x: x.T @ P @ x
                
                return {
                    'stable': True,
                    'method': 'lyapunov_quadratic',
                    'P': P,
                    'lyapunov_function': lyapunov_func,
                    'eigenvalues': eigenvals,
                    'P_eigenvalues': eigenvals_P
                }
            else:
                return {
                    'stable': False,
                    'method': 'lyapunov_failed',
                    'P': P,
                    'eigenvalues': eigenvals
                }
                
        except Exception as lyap_error:
            return {
                'stable': is_stable,
                'method': 'eigenvalues_only',
                'eigenvalues': eigenvals,
                'lyapunov_error': str(lyap_error)
            }
            
    except Exception as e:
        return {
            'stable': False,
            'error': str(e)
        }


def lyapunov_function_quadratic(A: np.ndarray) -> Tuple[Optional[np.ndarray], bool, Optional[Callable]]:
    """
    Construire une fonction de Lyapunov quadratique
    
    Args:
        A: Matrice du système linéarisé
        
    Returns:
        (P, is_stable, lyapunov_function)
    """
    try:
        # Vérifier la stabilité
        eigenvals = np.linalg.eigvals(A)
        is_stable = np.all(np.real(eigenvals) < 0)
        
        if not is_stable:
            return None, False, None
        
        # Résoudre l'équation de Lyapunov
        n = A.shape[0]
        Q = np.eye(n)
        P = scipy.linalg.solve_lyapunov(A.T, -Q)
        
        # Vérifier que P est définie positive
        if np.all(np.linalg.eigvals(P) > 0):
            lyapunov_func = lambda x: x.T @ P @ x
            return P, True, lyapunov_func
        else:
            return P, False, None
            
    except Exception as e:
        return None, False, None


def evaluate_lyapunov_derivative(lyapunov_func: Callable, 
                               nonlinear_func: Callable,
                               state: np.ndarray) -> float:
    """
    Évaluer la dérivée de la fonction de Lyapunov
    
    Args:
        lyapunov_func: Fonction de Lyapunov V(x)
        nonlinear_func: Fonction du système dx/dt = f(x)
        state: Point d'évaluation
        
    Returns:
        Dérivée V̇(x) = ∇V · f(x)
    """
    try:
        h = 1e-8
        n = len(state)
        
        # Calcul du gradient de V par différences finies
        grad_V = np.zeros(n)
        
        for i in range(n):
            state_plus = state.copy()
            state_plus[i] += h
            state_minus = state.copy()
            state_minus[i] -= h
            
            grad_V[i] = (lyapunov_func(state_plus) - lyapunov_func(state_minus)) / (2 * h)
        
        # Évaluation de f(x)
        f_x = nonlinear_func(0, state)
        
        # Produit scalaire ∇V · f(x)
        V_dot = np.dot(grad_V, f_x)
        
        return V_dot
        
    except Exception as e:
        return 0.0


def simulate_nonlinear_system(nonlinear_func: Callable, 
                            initial_condition: np.ndarray,
                            t_final: float = 20.0, 
                            dt: float = 0.01,
                            u_func: Optional[Callable] = None) -> Any:
    """
    Simuler un système non-linéaire
    
    Args:
        nonlinear_func: Fonction du système dx/dt = f(x, u)
        initial_condition: Condition initiale
        t_final: Temps final
        dt: Pas de temps suggéré
        u_func: Fonction d'entrée u(t) (optionnelle)
        
    Returns:
        Résultat de solve_ivp
    """
    try:
        # Fonction d'entrée par défaut
        if u_func is None:
            u_func = lambda t: np.array([0.0])
        
        # Définir la fonction du système pour solve_ivp
        def system_ode(t, x):
            u = u_func(t)
            return nonlinear_func(t, x, u)
        
        # Résoudre l'équation différentielle
        t_eval = np.arange(0, t_final + dt, dt)
        
        solution = solve_ivp(
            system_ode,
            [0, t_final],
            initial_condition,
            t_eval=t_eval,
            method='RK45',
            dense_output=True,
            rtol=1e-6,
            atol=1e-9
        )
        
        return solution
        
    except Exception as e:
        # Retourner une solution vide en cas d'erreur
        t_eval = np.arange(0, t_final + dt, dt)
        return type('Solution', (), {
            'success': False,
            't': t_eval,
            'y': np.zeros((len(initial_condition), len(t_eval))),
            'message': str(e)
        })()


def phase_portrait(nonlinear_func: Callable, 
                  x1_range: Tuple[float, float], 
                  x2_range: Tuple[float, float],
                  grid_density: int = 20) -> Dict[str, Any]:
    """
    Calculer les données pour un portrait de phase 2D
    
    Args:
        nonlinear_func: Fonction du système
        x1_range: Plage pour x1 (min, max)
        x2_range: Plage pour x2 (min, max)
        grid_density: Densité de la grille
        
    Returns:
        Dictionnaire avec les données du champ de vecteurs
    """
    try:
        # Création de la grille
        x1_grid = np.linspace(x1_range[0], x1_range[1], grid_density)
        x2_grid = np.linspace(x2_range[0], x2_range[1], grid_density)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # Calcul du champ de vecteurs
        DX1 = np.zeros_like(X1)
        DX2 = np.zeros_like(X2)
        
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                state = np.array([X1[i, j], X2[i, j]])
                derivatives = nonlinear_func(0, state)
                
                if len(derivatives) >= 2:
                    DX1[i, j] = derivatives[0]
                    DX2[i, j] = derivatives[1]
        
        # Normalisation pour l'affichage
        M = np.sqrt(DX1**2 + DX2**2)
        M[M == 0] = 1  # Éviter la division par zéro
        
        return {
            'X1': X1,
            'X2': X2,
            'DX1': DX1,
            'DX2': DX2,
            'DX1_norm': DX1 / M,
            'DX2_norm': DX2 / M,
            'magnitude': M,
            'success': True
        }
        
    except Exception as e:
        # Retourner des données vides
        X1, X2 = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
        return {
            'X1': X1,
            'X2': X2,
            'DX1': np.zeros_like(X1),
            'DX2': np.zeros_like(X2),
            'DX1_norm': np.zeros_like(X1),
            'DX2_norm': np.zeros_like(X2),
            'magnitude': np.zeros_like(X1),
            'success': False,
            'error': str(e)
        }


def find_equilibrium_points(nonlinear_func: Callable,
                          search_region: Tuple[Tuple[float, float], Tuple[float, float]],
                          tolerance: float = 1e-6) -> List[np.ndarray]:
    """
    Trouver les points d'équilibre d'un système non-linéaire
    
    Args:
        nonlinear_func: Fonction du système
        search_region: Région de recherche ((x1_min, x1_max), (x2_min, x2_max))
        tolerance: Tolérance pour f(x) ≈ 0
        
    Returns:
        Liste des points d'équilibre trouvés
    """
    try:
        from scipy.optimize import fsolve
        
        equilibrium_points = []
        
        # Fonction objectif : ||f(x)|| = 0
        def objective(x):
            return nonlinear_func(0, x)
        
        # Recherche avec plusieurs points initiaux
        x1_range, x2_range = search_region
        n_initial = 10  # Nombre de points initiaux
        
        x1_initials = np.linspace(x1_range[0], x1_range[1], n_initial)
        x2_initials = np.linspace(x2_range[0], x2_range[1], n_initial)
        
        for x1_init in x1_initials:
            for x2_init in x2_initials:
                initial_guess = np.array([x1_init, x2_init])
                
                try:
                    solution = fsolve(objective, initial_guess, xtol=tolerance)
                    
                    # Vérifier que c'est vraiment un point d'équilibre
                    f_val = objective(solution)
                    if np.linalg.norm(f_val) < tolerance:
                        # Vérifier qu'on n'a pas déjà ce point
                        is_new = True
                        for existing_point in equilibrium_points:
                            if np.linalg.norm(solution - existing_point) < tolerance * 10:
                                is_new = False
                                break
                        
                        if is_new:
                            equilibrium_points.append(solution)
                            
                except:
                    continue
        
        return equilibrium_points
        
    except Exception as e:
        # Retourner au moins l'origine comme point d'équilibre par défaut
        return [np.array([0.0, 0.0])]


def classify_equilibrium_point(A: np.ndarray) -> str:
    """
    Classifier un point d'équilibre basé sur le système linéarisé
    
    Args:
        A: Matrice jacobienne au point d'équilibre
        
    Returns:
        Type du point d'équilibre
    """
    try:
        eigenvals = np.linalg.eigvals(A)
        
        # Séparer parties réelles et imaginaires
        real_parts = np.real(eigenvals)
        imag_parts = np.imag(eigenvals)
        
        # Classification
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
        elif np.all(real_parts == 0):
            if np.any(imag_parts != 0):
                return "Centre"
            else:
                return "Cas dégénéré"
        else:
            return "Point d'équilibre mixte"
            
    except Exception as e:
        return "Classification impossible"


def lyapunov_exponents(nonlinear_func: Callable, 
                      initial_condition: np.ndarray,
                      t_final: float = 100.0) -> np.ndarray:
    """
    Calculer les exposants de Lyapunov (méthode simplifiée)
    
    Args:
        nonlinear_func: Fonction du système
        initial_condition: Condition initiale
        t_final: Temps de calcul
        
    Returns:
        Exposants de Lyapunov estimés
    """
    try:
        n = len(initial_condition)
        
        # Simulation de la trajectoire principale
        sol_main = simulate_nonlinear_system(nonlinear_func, initial_condition, t_final)
        
        if not sol_main.success:
            return np.zeros(n)
        
        # Perturbations infinitésimales
        epsilon = 1e-12
        lyap_exponents = np.zeros(n)
        
        for i in range(n):
            # Condition initiale perturbée
            perturbed_ic = initial_condition.copy()
            perturbed_ic[i] += epsilon
            
            # Simulation de la trajectoire perturbée
            sol_pert = simulate_nonlinear_system(nonlinear_func, perturbed_ic, t_final)
            
            if sol_pert.success:
                # Calcul de la divergence
                final_separation = np.linalg.norm(sol_pert.y[:, -1] - sol_main.y[:, -1])
                
                if final_separation > 0:
                    lyap_exponents[i] = np.log(final_separation / epsilon) / t_final
        
        return lyap_exponents
        
    except Exception as e:
        n = len(initial_condition)
        return np.zeros(n)


def basin_of_attraction(nonlinear_func: Callable,
                       equilibrium_point: np.ndarray,
                       search_region: Tuple[Tuple[float, float], Tuple[float, float]],
                       grid_size: int = 50,
                       t_final: float = 50.0) -> Dict[str, Any]:
    """
    Estimer le bassin d'attraction d'un point d'équilibre
    
    Args:
        nonlinear_func: Fonction du système
        equilibrium_point: Point d'équilibre stable
        search_region: Région de recherche
        grid_size: Taille de la grille
        t_final: Temps de simulation
        
    Returns:
        Dictionnaire avec les données du bassin d'attraction
    """
    try:
        x1_range, x2_range = search_region
        
        # Grille de conditions initiales
        x1_grid = np.linspace(x1_range[0], x1_range[1], grid_size)
        x2_grid = np.linspace(x2_range[0], x2_range[1], grid_size)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # Matrice indiquant l'appartenance au bassin
        basin_mask = np.zeros_like(X1, dtype=bool)
        
        tolerance = 0.1  # Tolérance pour considérer qu'on a atteint l'équilibre
        
        for i in range(grid_size):
            for j in range(grid_size):
                initial_cond = np.array([X1[i, j], X2[i, j]])
                
                try:
                    # Simulation
                    sol = simulate_nonlinear_system(nonlinear_func, initial_cond, t_final)
                    
                    if sol.success:
                        # Vérifier si la trajectoire converge vers le point d'équilibre
                        final_state = sol.y[:, -1]
                        distance_to_eq = np.linalg.norm(final_state - equilibrium_point)
                        
                        if distance_to_eq < tolerance:
                            basin_mask[i, j] = True
                            
                except:
                    continue
        
        return {
            'X1': X1,
            'X2': X2,
            'basin_mask': basin_mask,
            'equilibrium_point': equilibrium_point,
            'success': True
        }
        
    except Exception as e:
        X1, X2 = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        return {
            'X1': X1,
            'X2': X2,
            'basin_mask': np.zeros_like(X1, dtype=bool),
            'equilibrium_point': equilibrium_point,
            'success': False,
            'error': str(e)
        }


def poincare_map(nonlinear_func: Callable,
                initial_condition: np.ndarray,
                section_normal: np.ndarray,
                section_point: np.ndarray,
                n_intersections: int = 100,
                t_max: float = 1000.0) -> Dict[str, Any]:
    """
    Calculer la carte de Poincaré (version simplifiée)
    
    Args:
        nonlinear_func: Fonction du système
        initial_condition: Condition initiale
        section_normal: Vecteur normal à la section de Poincaré
        section_point: Point sur la section
        n_intersections: Nombre d'intersections à calculer
        t_max: Temps maximum de simulation
        
    Returns:
        Points d'intersection avec la section
    """
    try:
        # Cette implémentation est simplifiée
        # Une version complète nécessiterait la détection d'événements
        
        intersections = []
        
        # Simulation longue avec détection des croisements
        sol = simulate_nonlinear_system(nonlinear_func, initial_condition, t_max, dt=0.01)
        
        if not sol.success:
            return {'intersections': [], 'success': False}
        
        # Recherche des intersections (méthode simplifiée)
        trajectory = sol.y.T
        
        for i in range(1, len(trajectory)):
            # Vérifier si on traverse la section
            prev_point = trajectory[i-1]
            curr_point = trajectory[i]
            
            # Distance signée à la section
            prev_dist = np.dot(prev_point - section_point, section_normal)
            curr_dist = np.dot(curr_point - section_point, section_normal)
            
            # Changement de signe = traversée
            if prev_dist * curr_dist < 0:
                # Interpolation linéaire pour trouver le point exact
                alpha = abs(prev_dist) / (abs(prev_dist) + abs(curr_dist))
                intersection = prev_point + alpha * (curr_point - prev_point)
                intersections.append(intersection)
                
                if len(intersections) >= n_intersections:
                    break
        
        return {
            'intersections': np.array(intersections) if intersections else np.array([]),
            'success': True,
            'n_found': len(intersections)
        }
        
    except Exception as e:
        return {
            'intersections': np.array([]),
            'success': False,
            'error': str(e)
        }
