import { join, resolve } from 'path';
import { existsSync } from 'fs';
import getPort, { portNumbers } from 'get-port';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';

function addSecurityHeaders(server) {
  server.addHook('onSend', (request, reply, payload, done) => {
    reply.header('Cross-Origin-Opener-Policy', 'same-origin');
    reply.header('Cross-Origin-Embedder-Policy', 'require-corp');
    reply.header('Origin-Agent-Cluster', '?1');
    done();
  });
}

export async function startServer({ staticDir, modelsDir }) {
  const app = Fastify({ logger: false });
  addSecurityHeaders(app);

  const resolvedStatic = resolve(staticDir);
  app.register(fastifyStatic, {
    root: resolvedStatic,
    prefix: '/src/',
  });

  if (modelsDir && existsSync(modelsDir)) {
    app.register(fastifyStatic, {
      root: resolve(modelsDir),
      prefix: '/models/',
      decorateReply: false,
    });
  }

  app.get('/favicon.ico', async (_, reply) => reply.code(204).send());

  app.get('/', async (_, reply) => {
    return reply.sendFile('index.html');
  });

  const port = await getPort({ port: portNumbers(3000, 4000) });
  await app.listen({ port, host: '127.0.0.1' });
  const baseUrl = `http://127.0.0.1:${port}`;

  const close = async () => {
    try {
      await app.close();
    } catch (err) {
      app.log?.error?.(err);
    }
  };

  return { app, port, baseUrl, close };
}
