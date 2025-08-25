#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de contrôlabilité et observabilité
Calcul des matrices de contrôlabilité et observabilité, tests de rang
"""

import numpy as np
from typing import Tuple, Dict, Any
import scipy.linalg


def ctrb_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculer la matrice de contrôlabilité
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        
    Returns:
        Matrice de contrôlabilité Wc = [B AB A²B ... A^(n-1)B]
    """
    n = A.shape[0]
    m = B.shape[1]
    
    # Initialisation de la matrice de contrôlabilité
    Wc = np.zeros((n, n * m))
    
    # Calcul des puissances successives de A
    A_power = np.eye(n)
    
    for i in range(n):
        # Wc[:, i*m:(i+1)*m] = A^i @ B
        Wc[:, i*m:(i+1)*m] = A_power @ B
        A_power = A_power @ A
    
    return Wc


def obsv_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculer la matrice d'observabilité
    
    Args:
        A: Matrice d'état (n×n)
        C: Matrice de sortie (p×n)
        
    Returns:
        Matrice d'observabilité Wo = [C; CA; CA²; ...; CA^(n-1)]
    """
    n = A.shape[0]
    p = C.shape[0]
    
    # Initialisation de la matrice d'observabilité
    Wo = np.zeros((n * p, n))
    
    # Calcul des puissances successives de A
    A_power = np.eye(n)
    
    for i in range(n):
        # Wo[i*p:(i+1)*p, :] = C @ A^i
        Wo[i*p:(i+1)*p, :] = C @ A_power
        A_power = A_power @ A
    
    return Wo


def is_controllable(A: np.ndarray, B: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Tester la contrôlabilité d'un système
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        tolerance: Seuil de tolérance numérique
        
    Returns:
        True si le système est contrôlable, False sinon
    """
    try:
        n = A.shape[0]
        Wc = ctrb_matrix(A, B)
        rank_Wc = np.linalg.matrix_rank(Wc, tol=tolerance)
        
        return rank_Wc == n
        
    except Exception as e:
        return False


def is_observable(A: np.ndarray, C: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Tester l'observabilité d'un système
    
    Args:
        A: Matrice d'état (n×n)
        C: Matrice de sortie (p×n)
        tolerance: Seuil de tolérance numérique
        
    Returns:
        True si le système est observable, False sinon
    """
    try:
        n = A.shape[0]
        Wo = obsv_matrix(A, C)
        rank_Wo = np.linalg.matrix_rank(Wo, tol=tolerance)
        
        return rank_Wo == n
        
    except Exception as e:
        return False


def controllability_decomposition(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Décomposition contrôlable/non-contrôlable (décomposition de Kalman)
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        
    Returns:
        Dictionnaire avec les matrices transformées et les indices
    """
    try:
        n = A.shape[0]
        Wc = ctrb_matrix(A, B)
        
        # Décomposition SVD de la matrice de contrôlabilité
        U, s, Vt = np.linalg.svd(Wc, full_matrices=True)
        
        # Détermination du rang et de la dimension contrôlable
        rank_tol = 1e-10
        rank_Wc = np.sum(s > rank_tol)
        
        # Matrice de transformation
        T = U.T
        T_inv = U
        
        # Transformation du système
        A_transformed = T @ A @ T_inv
        B_transformed = T @ B
        
        # Partition des matrices
        A11 = A_transformed[:rank_Wc, :rank_Wc]  # Partie contrôlable
        A12 = A_transformed[:rank_Wc, rank_Wc:]  # Couplage
        A21 = A_transformed[rank_Wc:, :rank_Wc]  # Doit être zéro
        A22 = A_transformed[rank_Wc:, rank_Wc:]  # Partie non-contrôlable
        
        B1 = B_transformed[:rank_Wc, :]  # Entrée de la partie contrôlable
        B2 = B_transformed[rank_Wc:, :]  # Doit être zéro
        
        return {
            'controllable': rank_Wc == n,
            'controllable_dimension': rank_Wc,
            'total_dimension': n,
            'transformation_matrix': T,
            'A_transformed': A_transformed,
            'B_transformed': B_transformed,
            'A11': A11,  # Sous-système contrôlable
            'A12': A12,
            'A21': A21,
            'A22': A22,  # Sous-système non-contrôlable
            'B1': B1,
            'B2': B2,
            'singular_values': s
        }
        
    except Exception as e:
        return {
            'controllable': False,
            'error': str(e)
        }


def observability_decomposition(A: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
    """
    Décomposition observable/non-observable
    
    Args:
        A: Matrice d'état (n×n)
        C: Matrice de sortie (p×n)
        
    Returns:
        Dictionnaire avec les matrices transformées et les indices
    """
    try:
        n = A.shape[0]
        Wo = obsv_matrix(A, C)
        
        # Décomposition SVD de la matrice d'observabilité
        U, s, Vt = np.linalg.svd(Wo, full_matrices=True)
        
        # Détermination du rang et de la dimension observable
        rank_tol = 1e-10
        rank_Wo = np.sum(s > rank_tol)
        
        # Matrice de transformation (utiliser Vt pour les colonnes)
        T = Vt
        T_inv = Vt.T
        
        # Transformation du système
        A_transformed = T @ A @ T_inv
        C_transformed = C @ T_inv
        
        # Partition des matrices
        A11 = A_transformed[:rank_Wo, :rank_Wo]  # Partie observable
        A12 = A_transformed[:rank_Wo, rank_Wo:]  # Couplage
        A21 = A_transformed[rank_Wo:, :rank_Wo]  # Doit être zéro
        A22 = A_transformed[rank_Wo:, rank_Wo:]  # Partie non-observable
        
        C1 = C_transformed[:, :rank_Wo]  # Sortie de la partie observable
        C2 = C_transformed[:, rank_Wo:]  # Doit être zéro
        
        return {
            'observable': rank_Wo == n,
            'observable_dimension': rank_Wo,
            'total_dimension': n,
            'transformation_matrix': T,
            'A_transformed': A_transformed,
            'C_transformed': C_transformed,
            'A11': A11,  # Sous-système observable
            'A12': A12,
            'A21': A21,
            'A22': A22,  # Sous-système non-observable
            'C1': C1,
            'C2': C2,
            'singular_values': s
        }
        
    except Exception as e:
        return {
            'observable': False,
            'error': str(e)
        }


def kalman_decomposition(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
    """
    Décomposition de Kalman complète (contrôlable/observable, etc.)
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        C: Matrice de sortie (p×n)
        
    Returns:
        Dictionnaire avec la décomposition complète
    """
    try:
        n = A.shape[0]
        
        # Étape 1: Décomposition contrôlable/non-contrôlable
        ctrb_decomp = controllability_decomposition(A, B)
        
        if not ctrb_decomp.get('controllable', False):
            # Système partiellement contrôlable
            T1 = ctrb_decomp['transformation_matrix']
            A1 = ctrb_decomp['A_transformed']
            B1 = ctrb_decomp['B_transformed']
            C1 = C @ T1.T  # Transformer C aussi
            
            nc = ctrb_decomp['controllable_dimension']
        else:
            # Système complètement contrôlable
            T1 = np.eye(n)
            A1, B1, C1 = A, B, C
            nc = n
        
        # Étape 2: Décomposition observable/non-observable sur le système transformé
        obsv_decomp = observability_decomposition(A1, C1)
        
        T2 = obsv_decomp['transformation_matrix']
        A2 = obsv_decomp['A_transformed']
        C2 = obsv_decomp['C_transformed']
        B2 = T2 @ B1
        
        no = obsv_decomp['observable_dimension']
        
        # Transformation totale
        T_total = T2 @ T1
        
        # Partition finale en 4 blocs
        # [A_co  A_co_nco]  États contrôlables et observables (co)
        # [0     A_nco   ]  États contrôlables mais non observables (nco)
        # [0     0       ]  États non contrôlables mais observables (noc)
        # [0     0       ]  États non contrôlables et non observables (ncnoc)
        
        # Dimensions des sous-espaces
        n_co = min(nc, no)      # Contrôlable et observable
        n_nco = nc - n_co       # Contrôlable mais non observable
        n_noc = no - n_co       # Non contrôlable mais observable
        n_ncnoc = n - nc - n_noc # Non contrôlable et non observable
        
        return {
            'transformation_matrix': T_total,
            'A_decomposed': A2,
            'B_decomposed': B2,
            'C_decomposed': C2,
            'dimensions': {
                'controllable_observable': n_co,
                'controllable_nonobservable': n_nco,
                'noncontrollable_observable': n_noc,
                'noncontrollable_nonobservable': n_ncnoc
            },
            'controllable': nc == n,
            'observable': no == n,
            'controllable_dimension': nc,
            'observable_dimension': no,
            'total_dimension': n
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'controllable': False,
            'observable': False
        }


def controllability_gramian(A: np.ndarray, B: np.ndarray, method: str = 'lyapunov') -> np.ndarray:
    """
    Calculer le grammien de contrôlabilité
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        method: Méthode de calcul ('lyapunov' ou 'integration')
        
    Returns:
        Grammien de contrôlabilité Gc
    """
    try:
        if method == 'lyapunov':
            # Résolution de l'équation de Lyapunov: A*Gc + Gc*A^T + B*B^T = 0
            try:
                Gc = scipy.linalg.solve_lyapunov(A, -B @ B.T)
                return Gc
            except:
                # Fallback vers la méthode d'intégration
                method = 'integration'
        
        if method == 'integration':
            # Méthode d'intégration numérique: Gc = ∫₀^∞ e^(At) B B^T e^(A^T t) dt
            n = A.shape[0]
            Gc = np.zeros((n, n))
            
            # Paramètres d'intégration
            dt = 0.01
            t_max = 50.0  # Approximation de l'infini
            t_points = np.arange(0, t_max, dt)
            
            for t in t_points:
                eAt = scipy.linalg.expm(A * t)
                Gc += eAt @ B @ B.T @ eAt.T * dt
            
            return Gc
            
    except Exception as e:
        # Retourner une matrice identité en cas d'erreur
        return np.eye(A.shape[0])


def observability_gramian(A: np.ndarray, C: np.ndarray, method: str = 'lyapunov') -> np.ndarray:
    """
    Calculer le grammien d'observabilité
    
    Args:
        A: Matrice d'état (n×n)
        C: Matrice de sortie (p×n)
        method: Méthode de calcul ('lyapunov' ou 'integration')
        
    Returns:
        Grammien d'observabilité Go
    """
    try:
        if method == 'lyapunov':
            # Résolution de l'équation de Lyapunov: A^T*Go + Go*A + C^T*C = 0
            try:
                Go = scipy.linalg.solve_lyapunov(A.T, -C.T @ C)
                return Go
            except:
                # Fallback vers la méthode d'intégration
                method = 'integration'
        
        if method == 'integration':
            # Méthode d'intégration numérique: Go = ∫₀^∞ e^(A^T t) C^T C e^(At) dt
            n = A.shape[0]
            Go = np.zeros((n, n))
            
            # Paramètres d'intégration
            dt = 0.01
            t_max = 50.0  # Approximation de l'infini
            t_points = np.arange(0, t_max, dt)
            
            for t in t_points:
                eAt = scipy.linalg.expm(A.T * t)
                Go += eAt @ C.T @ C @ eAt.T * dt
            
            return Go
            
    except Exception as e:
        # Retourner une matrice identité en cas d'erreur
        return np.eye(A.shape[0])


def hankel_singular_values(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculer les valeurs singulières de Hankel
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        C: Matrice de sortie (p×n)
        
    Returns:
        Valeurs singulières de Hankel (triées par ordre décroissant)
    """
    try:
        # Calcul des grammiens
        Gc = controllability_gramian(A, B)
        Go = observability_gramian(A, C)
        
        # Les valeurs singulières de Hankel sont les racines carrées 
        # des valeurs propres du produit Gc*Go
        eigenvals = np.linalg.eigvals(Gc @ Go)
        
        # Filtrer les valeurs propres négatives (erreurs numériques)
        eigenvals = eigenvals[eigenvals >= 0]
        
        # Valeurs singulières de Hankel
        hsv = np.sqrt(eigenvals)
        
        # Trier par ordre décroissant
        hsv = np.sort(hsv)[::-1]
        
        return hsv
        
    except Exception as e:
        return np.array([])


def minimal_realization_indices(A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                               tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Calculer les indices pour une réalisation minimale
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        C: Matrice de sortie (p×n)
        tolerance: Seuil de tolérance numérique
        
    Returns:
        Dictionnaire avec les indices de réalisation minimale
    """
    try:
        n = A.shape[0]
        
        # Contrôlabilité et observabilité
        is_ctrb = is_controllable(A, B, tolerance)
        is_obsv = is_observable(A, C, tolerance)
        
        # Rangs des matrices
        Wc = ctrb_matrix(A, B)
        Wo = obsv_matrix(A, C)
        rank_ctrb = np.linalg.matrix_rank(Wc, tol=tolerance)
        rank_obsv = np.linalg.matrix_rank(Wo, tol=tolerance)
        
        # Dimension minimale
        minimal_dimension = min(rank_ctrb, rank_obsv)
        
        # Valeurs singulières de Hankel
        hsv = hankel_singular_values(A, B, C)
        
        # États significatifs (basés sur les valeurs singulières de Hankel)
        significant_states = np.sum(hsv > tolerance)
        
        return {
            'original_dimension': n,
            'minimal_dimension': minimal_dimension,
            'controllable_dimension': rank_ctrb,
            'observable_dimension': rank_obsv,
            'significant_states_hankel': significant_states,
            'is_controllable': is_ctrb,
            'is_observable': is_obsv,
            'is_minimal': is_ctrb and is_obsv and n == minimal_dimension,
            'reduction_possible': n > minimal_dimension,
            'hankel_singular_values': hsv,
            'reduction_ratio': minimal_dimension / n if n > 0 else 0
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_minimal': False,
            'reduction_possible': False
        }


def controllable_subspace_basis(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculer une base pour le sous-espace contrôlable
    
    Args:
        A: Matrice d'état (n×n)
        B: Matrice d'entrée (n×m)
        
    Returns:
        Matrice dont les colonnes forment une base du sous-espace contrôlable
    """
    try:
        Wc = ctrb_matrix(A, B)
        
        # Décomposition QR pour obtenir une base orthonormale
        Q, R = np.linalg.qr(Wc)
        
        # Déterminer le rang
        rank_tol = 1e-10
        rank = np.sum(np.abs(np.diag(R)) > rank_tol)
        
        # Retourner les premières colonnes de Q correspondant au rang
        return Q[:, :rank]
        
    except Exception as e:
        # Retourner une base triviale
        return np.eye(A.shape[0], 1)


def observable_subspace_basis(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculer une base pour le sous-espace observable
    
    Args:
        A: Matrice d'état (n×n)
        C: Matrice de sortie (p×n)
        
    Returns:
        Matrice dont les colonnes forment une base du sous-espace observable
    """
    try:
        Wo = obsv_matrix(A, C)
        
        # Décomposition QR sur la transposée pour obtenir une base des colonnes
        Q, R = np.linalg.qr(Wo.T)
        
        # Déterminer le rang
        rank_tol = 1e-10
        rank = np.sum(np.abs(np.diag(R)) > rank_tol)
        
        # Retourner les premières colonnes de Q correspondant au rang
        return Q[:, :rank]
        
    except Exception as e:
        # Retourner une base triviale
        return np.eye(A.shape[0], 1)
