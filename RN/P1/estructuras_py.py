'''MANEJO DE ESTRUCTURAS EN PYTHON 
6BV1 
EQUIPO 6 
CODIGO ELABORADO POR 
MONTERO BARRAZA ALVARO 
GARCIA LOPEZ LUIS 

'''

dictionary={"Neza":(5,0),"Aeropuerto":(1,0),"ESCOM":(-2,4),"Xochi":(2,-5)} #USO DE DICCIONARIO
lugares=["Neza","Aeropuerto","ESCOM","Xochi"] #USO DE LISTA 

def calc_dist(pos:tuple,key): #FUNCION PARA CALCULAR LA DISTANCIA USANDO LOS INDICES DE LA LISTA DE LUGARES
    tup1=dictionary[lugares[key]] #ACCESO A LA TUPLA DEL DICCIONARIO
    return (((pos[0]-tup1[0])**2+(pos[1]-tup1[1])**2)**.5,((pos[0]-tup1[0])+(pos[1]-tup1[1]))) #RETORNAMOS EL CALCULO



cadena=input("Introduce tus coordenadas actuales\n") #CREACION DE CADENA 
cadena=(cadena.split(",",1)) #DIVISION DE LA CADENA POR COMAS 
x=int(cadena[0]) #CONVERSION DE CADENA A INT 
y=int(cadena[1])
print(lugares)
key=int(input("Introduce el lugar, empezando por 0\n")) #LECTURA DE INPUT 
euc,manh=calc_dist((x,y),key) #LLAMADA DE LA FUNCION QUE SE DESCOMPONE EN UNA TUPLA 
print("Estás a "+str(euc)+" km lineales, pero a "+str(manh)+" en distancia real") #DISPLAY DE RESULTADOS

