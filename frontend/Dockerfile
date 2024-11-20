FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install --no-audit --no-fund --no-optional

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
