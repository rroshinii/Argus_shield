const localtunnel = require('localtunnel');

(async () => {
  try {
    const tunnel = await localtunnel({ port: 8000, local_host: '127.0.0.1' });
    console.log('PUBLIC_URL:' + tunnel.url);
    
    tunnel.on('close', () => console.log('Tunnel closed'));
    tunnel.on('error', (err) => console.error('Tunnel error:', err));
    setInterval(() => {}, 10000);
  } catch (err) {
    console.error('Failed to start tunnel:', err);
    process.exit(1);
  }
})();
