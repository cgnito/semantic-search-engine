const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export interface TweetResult {
  text: string;
  date: string;
}

export async function searchTweets(query: string): Promise<TweetResult[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) throw new Error("Network response was not ok");
    return await response.json();
  } catch (error) {
    console.error("Search API Error:", error);
    return [];
  }
}