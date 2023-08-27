{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN4vhbQ5jGDljp5qEHQFyRm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/annu96720/Hand-Written-Digit-Predection/blob/main/Written_Prediction.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "8XnjVBZbqyhC"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np"
      ],
      "metadata": {
        "id": "F_D5TaWhtF8f"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "vcbYcBkjtXh9"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import load_digits"
      ],
      "metadata": {
        "id": "86eRaHf_tbcU"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = load_digits()"
      ],
      "metadata": {
        "id": "oy6Cx80vthDS"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))\n",
        "for ax, image, label in zip(axes, df.images, df.target):\n",
        "    ax.set_axis_off()\n",
        "    ax.imshow(image, cmap=plt.cm.gray_r, interpolation=\"nearest\")\n",
        "    ax.set_title(\"Training: %i\" % label)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 227
        },
        "id": "kxaVrx6FtmOy",
        "outputId": "93e1f6d9-8502-4d98-a6f5-20c9b3593969"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x300 with 4 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAxsAAADSCAYAAAAi0d0oAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAASFklEQVR4nO3db5CVZd0H8N8KsRsBsiLkkiUsOmPJIAHNJCbgsBCkBkmgLxhZxgYqGaM/M8sU5oJlkjZjhRnxBgNzlDLIJlMY2JymN7GyloYzSyw6GU6Kyx9F/no/L57HfaIld8Hr8rC7n88MM+x1zv29rwP82POd++w5ZUVRFAEAAJDYOaXeAAAA0D0pGwAAQBbKBgAAkIWyAQAAZKFsAAAAWSgbAABAFsoGAACQhbIBAABkoWwAAABZKBtnoLa2NoYNG3ZGx9bX10dZWVnaDcFZyJxAx8wJdMycdG3dqmyUlZV16ldDQ0Opt3rW+dOf/hSf+tSnom/fvnHBBRfErbfeGq+//nqpt0UG5uTMPPnkk3HzzTfHyJEjo1evXmf8jY+uwZycvkOHDsV9990XU6dOjaqqqujfv398/OMfj/vvvz9OnDhR6u2RgTk5M3feeWd88pOfjMGDB0dFRUVccsklsXjx4njllVdKvbUsyoqiKEq9iVTWrVt30tc///nPY9OmTbF27dqT1qdMmRIf/OAHz/g8x44di7feeivKy8tP+9jjx4/H8ePHo6Ki4ozPn1pTU1NcccUV8dGPfjQWLFgQ//jHP+Kee+6Jq6++Oh5//PFSb4/EzMmZqa2tjYcffjjGjBkTL774YvTq1St2795d6m2RiTk5fc8++2yMGjUqJk+eHFOnTo0BAwbEE088Eb/+9a/jpptuigceeKDUWyQxc3JmZs2aFYMHD45LL700+vfvHzt27IjVq1fHkCFDoqmpKT7wgQ+UeotpFd3YLbfcUnTmIb7xxhvvwW7OXtOnTy+qqqqK/fv3t62tXr26iIjiiSeeKOHOeC+Yk8556aWXiqNHjxZFURTXXHNNcdFFF5V2Q7ynzEnHXnnlleLZZ59ttz5//vwiIorm5uYS7Ir3kjk5c7/85S+LiCgeeuihUm8luW71MqrOmDRpUowcOTIaGxtjwoQJ0bdv3/jmN78ZEREbN26Ma665JoYOHRrl5eUxYsSIuOOOO9pd/v3P1w7u3r07ysrK4p577omf/exnMWLEiCgvL49PfOIT8ec///mkY0/12sGysrJYtGhRbNiwIUaOHBnl5eVx2WWXxe9///t2+29oaIhx48ZFRUVFjBgxIlatWnXKzFdffTWef/75OHTo0Dv+eRw4cCA2bdoUc+fOjQEDBrSt33TTTdGvX7945JFH3vF4uidz0t7QoUPjfe97X4f3o+cwJyc7//zz47LLLmu3/rnPfS4iInbs2PGOx9M9mZPOefvx7du374yOP5v1LvUGSmHv3r0xffr0uPHGG2Pu3Lltl/bWrFkT/fr1i6997WvRr1+/2LJlS3z729+OAwcOxN13391h7i9+8Ys4ePBgLFy4MMrKyuL73/9+XH/99bFr164On6T88Y9/jEcffTS+/OUvR//+/eNHP/pRzJo1K1588cUYNGhQRERs3749pk2bFlVVVbFs2bI4ceJELF++PAYPHtwub+XKlbFs2bLYunVrTJo06b+e969//WscP348xo0bd9J6nz59YvTo0bF9+/YOHzfdkzmBjpmTjr388ssR8b9lhJ7JnLRXFEXs3bs3jh8/Hs3NzbFkyZLo1atX9/xeVOpLKzmd6nLexIkTi4gofvrTn7a7/6FDh9qtLVy4sOjbt29x+PDhtrV58+ad9BKKlpaWIiKKQYMGFa+99lrb+saNG4uIKB577LG2tdtvv73dniKi6NOnT7Fz5862tWeeeaaIiOLHP/5x29p1111X9O3bt3jppZfa1pqbm4vevXu3y3z7PFu3bm33mP7d+vXri4gonnrqqXa3zZ49u7jgggve8Xi6PnPS8Zz8Jy+j6nnMyenPSVEUxZEjR4qPfexjxfDhw4tjx46d9vF0Leak83OyZ8+eIiLafl144YXFww8/3Klju5oe9zKqiIjy8vKYP39+u/X3v//9bb8/ePBgvPrqq3HVVVfFoUOH4vnnn+8w94YbbojKysq2r6+66qqIiNi1a1eHx9bU1MSIESPavh41alQMGDCg7dgTJ07E5s2bY+bMmTF06NC2+1188cUxffr0dnn19fVRFEWHDfnNN9+MiDjlD11VVFS03U7PY06gY+bknS1atCj+9re/xcqVK6N37x75YgrCnJzKeeedF5s2bYrHHnssli9fHueff363fRfQHjn5H/rQh6JPnz7t1p977rlYunRpbNmyJQ4cOHDSbfv37+8w9yMf+chJ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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.images.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MN-Keg05tqvi",
        "outputId": "2d0325da-c678-4f00-8d8a-fc8339bf5142"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1797, 8, 8)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.images[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dt209Nyftvkb",
        "outputId": "b3d6201b-751b-4038-c2f2-6b94db363e3d"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],\n",
              "       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],\n",
              "       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],\n",
              "       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],\n",
              "       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],\n",
              "       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],\n",
              "       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],\n",
              "       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.images[0].shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Eyk0e341t0Mv",
        "outputId": "13998ffd-1489-42c7-99fe-4177b94c240d"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(8, 8)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "len(df.images)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xiYyxoqdt5xP",
        "outputId": "bf3f2be8-f1ca-4da1-a7de-65e53ae4b85e"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1797"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "n_samples = len(df.images)\n",
        "data = df.images.reshape((n_samples, -1))"
      ],
      "metadata": {
        "id": "rfj5Y7AGt94K"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2Hky3Ba8uD7p",
        "outputId": "877277da-ea5f-484e-96ba-1108efe927ab"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,\n",
              "       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,\n",
              "       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,\n",
              "        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,\n",
              "       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[0].shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YSkCABltuHre",
        "outputId": "99334b90-db32-4d07-87ba-dfb30ec33079"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(64,)"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cc-rawdWuNeb",
        "outputId": "78c46ef7-8bca-4efb-d909-cfc7b3e2726f"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1797, 64)"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.min()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JqL4mFmKuP8m",
        "outputId": "4faed26d-5c0d-47da-d67d-482c3eb7eff7"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.0"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.max()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zYe-qucZuUFb",
        "outputId": "949dc844-0994-43d1-8ace-79aecca0b484"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "16.0"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = data/16"
      ],
      "metadata": {
        "id": "6TavD8BUuctH"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.min()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "U9RYe-OHufny",
        "outputId": "9c68ada4-9fb7-4b1c-91ca-bc7e9c137efb"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.0"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.max()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DwdIULDUuii_",
        "outputId": "5e24ddbb-9c95-4c9a-fc79-c0b9df5c181d"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "huGW0ep4umqK",
        "outputId": "b240eec8-bd66-43b3-b56c-9c5765faab4f"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0.    , 0.    , 0.3125, 0.8125, 0.5625, 0.0625, 0.    , 0.    ,\n",
              "       0.    , 0.    , 0.8125, 0.9375, 0.625 , 0.9375, 0.3125, 0.    ,\n",
              "       0.    , 0.1875, 0.9375, 0.125 , 0.    , 0.6875, 0.5   , 0.    ,\n",
              "       0.    , 0.25  , 0.75  , 0.    , 0.    , 0.5   , 0.5   , 0.    ,\n",
              "       0.    , 0.3125, 0.5   , 0.    , 0.    , 0.5625, 0.5   , 0.    ,\n",
              "       0.    , 0.25  , 0.6875, 0.    , 0.0625, 0.75  , 0.4375, 0.    ,\n",
              "       0.    , 0.125 , 0.875 , 0.3125, 0.625 , 0.75  , 0.    , 0.    ,\n",
              "       0.    , 0.    , 0.375 , 0.8125, 0.625 , 0.    , 0.    , 0.    ])"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "wG1HSBAfupZx"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(data, df.target, test_size=0.3)"
      ],
      "metadata": {
        "id": "BJkGGSStuwDG"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BbR_70Xyuz9Z",
        "outputId": "f6d803bb-eb2f-42bb-f096-79ff8be225f8"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((1257, 64), (540, 64), (1257,), (540,))"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestClassifier"
      ],
      "metadata": {
        "id": "IMryW8Sxu5jm"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rf = RandomForestClassifier()"
      ],
      "metadata": {
        "id": "uW6_H_kDu8KW"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rf.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "AIo_BjyvvAuw",
        "outputId": "809b37a2-5fe8-4ebf-d8c0-557f8e757bb9"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = rf.predict(X_test)"
      ],
      "metadata": {
        "id": "BOsldv-kvFZx"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cYxd803uvJUL",
        "outputId": "3bfe3104-fa39-4f2d-efb6-844d252fd544"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2, 7, 2, 4, 6, 0, 5, 2, 1, 2, 9, 7, 6, 5, 9, 9, 8, 0, 4, 5, 4, 2,\n",
              "       8, 8, 3, 9, 3, 2, 7, 2, 5, 9, 4, 9, 6, 9, 2, 0, 1, 0, 3, 2, 6, 1,\n",
              "       2, 2, 4, 4, 9, 0, 2, 0, 7, 3, 5, 9, 5, 1, 6, 9, 1, 9, 7, 3, 6, 1,\n",
              "       5, 9, 9, 7, 5, 1, 1, 3, 7, 2, 6, 8, 0, 0, 3, 5, 2, 4, 2, 9, 2, 7,\n",
              "       1, 6, 0, 8, 6, 5, 1, 8, 1, 5, 4, 8, 1, 1, 6, 8, 5, 9, 4, 0, 1, 6,\n",
              "       2, 1, 4, 2, 9, 5, 7, 8, 2, 4, 1, 8, 7, 4, 6, 9, 3, 1, 7, 1, 9, 5,\n",
              "       0, 8, 1, 3, 1, 5, 5, 0, 3, 8, 3, 7, 0, 5, 4, 0, 9, 3, 4, 6, 9, 5,\n",
              "       9, 6, 9, 7, 0, 2, 0, 6, 3, 5, 3, 8, 7, 3, 8, 3, 9, 1, 2, 8, 9, 9,\n",
              "       5, 4, 3, 7, 4, 5, 6, 7, 1, 6, 1, 9, 4, 8, 8, 4, 7, 4, 3, 7, 9, 0,\n",
              "       3, 8, 3, 4, 7, 4, 4, 7, 0, 0, 6, 3, 8, 7, 2, 3, 6, 1, 4, 8, 6, 4,\n",
              "       1, 0, 6, 1, 1, 7, 1, 5, 1, 8, 8, 4, 9, 2, 1, 7, 2, 2, 7, 6, 4, 2,\n",
              "       9, 8, 1, 3, 8, 4, 2, 8, 9, 5, 6, 7, 7, 0, 6, 0, 5, 9, 9, 8, 3, 6,\n",
              "       9, 1, 7, 8, 2, 4, 7, 1, 6, 6, 6, 1, 0, 4, 2, 6, 2, 4, 2, 0, 9, 8,\n",
              "       5, 5, 1, 1, 0, 4, 6, 7, 8, 7, 7, 6, 6, 4, 6, 2, 0, 2, 4, 7, 0, 6,\n",
              "       3, 3, 9, 2, 5, 0, 3, 5, 9, 7, 4, 2, 9, 6, 7, 1, 7, 1, 4, 6, 3, 1,\n",
              "       4, 6, 6, 9, 6, 4, 3, 7, 1, 8, 5, 3, 8, 3, 0, 9, 4, 8, 4, 8, 3, 0,\n",
              "       9, 9, 0, 4, 9, 8, 7, 3, 2, 1, 6, 5, 5, 9, 9, 8, 7, 5, 6, 6, 7, 0,\n",
              "       2, 2, 3, 3, 7, 8, 5, 0, 4, 5, 8, 6, 9, 7, 9, 6, 0, 0, 2, 2, 9, 4,\n",
              "       9, 1, 0, 1, 6, 8, 5, 3, 9, 2, 3, 3, 3, 0, 9, 5, 1, 3, 8, 8, 1, 8,\n",
              "       6, 5, 3, 0, 8, 9, 9, 1, 6, 3, 6, 0, 3, 8, 3, 7, 9, 8, 6, 1, 5, 0,\n",
              "       1, 6, 6, 2, 8, 7, 5, 3, 3, 2, 5, 7, 4, 4, 4, 9, 4, 1, 8, 5, 8, 2,\n",
              "       1, 2, 2, 0, 1, 4, 3, 6, 5, 0, 6, 8, 7, 5, 8, 7, 7, 2, 4, 8, 9, 7,\n",
              "       6, 2, 8, 7, 8, 1, 3, 8, 7, 5, 2, 1, 5, 8, 2, 0, 4, 8, 5, 5, 0, 4,\n",
              "       5, 5, 4, 4, 8, 6, 4, 7, 3, 1, 5, 2, 7, 1, 0, 8, 4, 3, 5, 4, 4, 3,\n",
              "       2, 2, 8, 6, 2, 0, 1, 5, 3, 7, 4, 4])"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, classification_report"
      ],
      "metadata": {
        "id": "BAV-LiBBvPjt"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "confusion_matrix(y_test, y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sRZBoj-1vVJQ",
        "outputId": "5e57ad5a-deb6-4d1b-b6e5-6c72fe9c8e79"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[46,  0,  0,  0,  1,  0,  0,  0,  0,  0],\n",
              "       [ 0, 52,  0,  0,  0,  0,  0,  0,  0,  0],\n",
              "       [ 0,  0, 54,  0,  0,  0,  0,  0,  0,  0],\n",
              "       [ 0,  1,  0, 50,  0,  0,  0,  0,  4,  0],\n",
              "       [ 0,  0,  0,  0, 57,  0,  0,  0,  0,  0],\n",
              "       [ 0,  0,  0,  0,  0, 51,  0,  0,  0,  1],\n",
              "       [ 0,  0,  0,  0,  0,  1, 56,  0,  0,  0],\n",
              "       [ 0,  0,  0,  0,  0,  0,  0, 53,  0,  0],\n",
              "       [ 0,  3,  0,  1,  0,  0,  0,  0, 53,  1],\n",
              "       [ 0,  0,  0,  1,  0,  0,  0,  0,  1, 53]])"
            ]
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jG1Lj-5zvX6T",
        "outputId": "4fc57c57-75bc-4ce6-ff9e-8f8e4a4191dc"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.98      0.99        47\n",
            "           1       0.93      1.00      0.96        52\n",
            "           2       1.00      1.00      1.00        54\n",
            "           3       0.96      0.91      0.93        55\n",
            "           4       0.98      1.00      0.99        57\n",
            "           5       0.98      0.98      0.98        52\n",
            "           6       1.00      0.98      0.99        57\n",
            "           7       1.00      1.00      1.00        53\n",
            "           8       0.91      0.91      0.91        58\n",
            "           9       0.96      0.96      0.96        55\n",
            "\n",
            "    accuracy                           0.97       540\n",
            "   macro avg       0.97      0.97      0.97       540\n",
            "weighted avg       0.97      0.97      0.97       540\n",
            "\n"
          ]
        }
      ]
    }
  ]
}
