permutations = []

def get_permutations(i, k, l, temp):
    temp = temp*10 + i
    if k==1:
        permutations.append(temp)
    elif i<l:
        for j in range(i+1,l):
            get_permutations(j, k-1, l, temp)

get_permutations(0,3+1,5+1,0)

print("aa",permutations)