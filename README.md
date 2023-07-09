# Projeto de Tópicos em Bancos de Dados IC007
## :boom: Como rodar

Certifique-se de possuir o docker na sua máquina e uma base de dados firestore (gere uma chave privada para esta base). 

Clone o repositório e mude para raiz dele.

```sh
# Copie o .settings.toml
$ cp .settings.toml .settings.local.toml

```
Coloque o arquivo json com sua chave privada na mesma pasta do repositório, e insira o nome deste na variável de ambiente "json_name" em "CRED".

Build a imagem:

```sh
# Buildando a imagem
$ docker build -t yolov8_csis .
```

Rode a imagem em um container:

```sh
# Rode um container com base na imagem
$ docker run -p 7100:7100 -it yolov8_csis
```

Para acessar as docs do Swagger basta abrir http://localhost:7100/docs# no navegador.

Utilize alguma aplicação de requests e faça os devidos requests para o endereço base http://localhost:7100.
