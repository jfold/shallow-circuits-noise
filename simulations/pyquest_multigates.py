import numpy as np
from pyquest.unitaries import H, S, U, Z


def Rxx(q1, q2, theta) -> U:
    matrix = np.array(
        [
            [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
    )
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(4))
    gate = U(targets=[q1, q2], matrix=matrix)
    return gate


def Ryy(q1, q2, theta):
    matrix = np.array(
        [
            [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
        ]
    )
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(4))
    gate = U(targets=[q1, q2], matrix=matrix)
    return gate


def Rzz(q1, q2, theta):
    matrix = np.array(
        [
            [np.exp(-1j * theta / 2), 0, 0, 0],
            [0, np.exp(1j * theta / 2), 0, 0],
            [0, 0, np.exp(1j * theta / 2), 0],
            [0, 0, 0, np.exp(-1j * theta / 2)],
        ]
    )
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(4))
    gate = U(targets=[q1, q2], matrix=matrix)
    return gate


def Rx(q, theta):
    matrix = np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(2))
    gate = U(target=q, matrix=matrix)
    return gate


def Ry(q, theta):
    matrix = np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(2))
    gate = U(target=q, matrix=matrix)
    return gate


def Rz(q, theta):
    matrix = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    assert np.allclose(matrix @ matrix.T.conj(), np.eye(2))
    gate = U(target=q, matrix=matrix)
    return gate


def Z2X_cob(q):
    gate = H(target=q)
    return [gate]


def Z2Y_cob(q):
    gate = [Rx(q, -np.pi / 2)]
    return gate


def X2Z_cob(q):
    gate = H(target=q)
    return gate


def Y2Z_cob(q):
    gate = [H(q), S(q)]
    return gate
