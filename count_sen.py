from collections import defaultdict


def count(file):
    d = defaultdict(lambda: 0)
    with open(file, 'r') as f:
        for line in f:
            eles = line.split()
            if len(eles) > 1:
                d[eles[1]] += 1
    return d


if __name__ == '__main__':
    filename = './fil_l5.txt'
    result = count(filename)
    print('neu', result['neu'])
    print('ang', result['ang'])
    print('hap', result['hap'])
    print('sad', result['sad'])
