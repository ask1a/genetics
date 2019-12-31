'''
1/ Population de base générée aléatoirement
    n chaînes de caractères ou de bits.
    1 caractère correspond à 1 gène
2/ Évaluation
    à chaque chaîne, une note correspondant à son adaptation au problème.
3/ Sélection
    tirage au sort de n/2 couples de chaînes sur une roue biaisée.
    Chaque chaîne a une probabilité d’être tirée proportionnelle à son adaptation au problème.
    Optimisation possible : si l’individu le plus adapté n’a pas été sélectionné, il est copié d’office dans la génération intermédiaire à la place d’un individu choisi aléatoirement.
4/ Croisement et mutation
    Chaque couple donne 2 chaînes filles.
    Enjambement. Probabilité : 70 %. Emplacement de l'enjambement choisi aléatoirement.
    Exemple :
    Chaînes parents : A : 00110100 ; B : 01010010
    Chaînes filles : A’ : 00010010 ;B’ : 01110100
    Croisement en 2 points plus efficace.
    Mutations des chaînes filles. Probabilité : de 0,1 à 1 %.
    Inversion d’un bit au hasard ou remplacement au hasard d’un caractère par un autre.
    Probabilité fixe ou évolutive (auto-adaptation).
    On peut prendre probabilité = 1/nombre de bits.
'''
import numpy as np
import random


def generation_pop_base(taille_pop, nb_genes, proba_pop_base):
    return np.random.choice(a=[1, 0], size=(taille_pop, nb_genes), p=[proba_pop_base, 1 - proba_pop_base])


def evaluation_indiv(index_individu, pop):
    return sum(pop[index_individu])


def evaluation_pop(pop):
    return np.sum(pop, 1)


def tirage_competiteurs(n):
    l = list(range(n))
    random.shuffle(l)
    it = iter(l)
    return [e for e in zip(it, it)]


def duel(pop, indiv1, indiv2):
    if evaluation_indiv(indiv1, pop) >= evaluation_indiv(indiv2, pop):
        return indiv1
    else:
        return indiv2


def selection_tournoi(pop):
    matchs = tirage_competiteurs(len(pop))
    return np.array([duel(pop, e[0], e[1]) for e in matchs])


def croisement(pop, idx_indiv1, idx_indiv2, taux_croisement):
    longueur = len(pop[idx_indiv1])
    choix = np.random.choice(a=[True, False], size=(longueur,), p=[taux_croisement, 1 - taux_croisement])
    ll = np.zeros((longueur,), dtype='int')
    for i in range(len(pop[idx_indiv1])):
        if choix[i]:
            ll[i] = pop[idx_indiv2][i]
        else:
            ll[i] = pop[idx_indiv1][i]
    return ll


def reproduction(pop, selection, taux_croisement):
    couples = [e for e in zip(selection, np.roll(selection, 1))]
    enfants = np.array([croisement(pop, e[0], e[1], taux_croisement) for e in couples])
    return np.concatenate((pop[selection], enfants))


def mutation(pop, taux_mutation):
    pop2 = np.copy(pop)
    n, p = pop2.shape[0] * pop2.shape[1], taux_mutation
    s = int(np.random.binomial(n, p, 1))
    to_invert = random.sample(range(n), s)

    forme = pop2.shape
    population_flat = np.ravel(pop2)
    for e in to_invert:
        population_flat[e] = 1 - population_flat[e]
    return population_flat.reshape(forme)


def comparaison_pop(pop1, pop2):
    return abs(sum(np.mean(pop1, axis=0) - np.mean(pop2, axis=0)))


taille_pop = 1000
nb_genes = 50
proba_pop_base = 0.5

taux_croisement = 0.5
taux_mutation = 0.0001

critere_arret = 0.0

population = generation_pop_base(taille_pop, nb_genes, proba_pop_base)

iteration = 0
while True:
    iteration += 1
    selection = selection_tournoi(population)
    pop_next_gen = reproduction(population, selection, taux_croisement)
    pop_mutee = mutation(pop_next_gen, taux_mutation)

    # critere d arret
    print(comparaison_pop(population, pop_mutee))
    if comparaison_pop(population, pop_mutee) <= critere_arret:
        break

    population = pop_mutee

print(iteration)
print(population[:5])
