
const testQueries = [
  "What's the weather in New York and London?",
  "Summarize the following articles: https://www.wired.com/ and https://www.theverge.com/",
  "Get the top headlines from Hacker News and Reddit's /r/programming subreddit.",
  "Translate 'hello world' to both Spanish and French.",
];

async function evaluate(sendMessage) {
  const results = {};
  for (const query of testQueries) {
    const response = await sendMessage(query);
    // This is a simplified evaluation. A more robust solution would involve
    // inspecting the tool_calls object in the response to see if the calls
    // were made in parallel.
    const parallel = response.tool_calls && response.tool_calls.length > 1;
    results[query] = parallel ? "PASS" : "FAIL";
  }
  return results;
}

export { evaluate };
