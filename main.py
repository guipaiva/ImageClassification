from show_imgs import plot_imgs

def main():
	print('Digite o índice inicial das imagens a serem mostradas.')
	print('Um total de 9 imagens e suas predições aparecerão na tela.')
	i = int(input())
	plot_imgs(i)

main()